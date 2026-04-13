from smolgpt.model import GPT
from smolgpt.config import GPTConfig, TrainingConfig
from functools import partial
import time
import math
import os
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from smolgpt.dataset import Task
from smolgpt.tokenizer import Tokenizer
from contextlib import nullcontext

from pathlib import Path

# directorio
ROOT_DIR = Path(__file__).resolve().parents[2]

train_config = TrainingConfig()
out_dir = "out/"
writer = SummaryWriter(log_dir=os.path.join(out_dir, "logs"))
# Como NO quiero entrenar desde el dataset de tinystories pongo el resume a False, si quiese continuar el entrenamiendo desde un checkpoing lo ponemos a True y partiria desde ckpt.pt
resume = False
ddp = int(os.environ.get("RANK", -1)) != -1
tokenizer = Tokenizer(str(ROOT_DIR / "data" / f"tok{GPTConfig.vocab_size}.model"))

# Preguntas fijas para evaluar generación real, así no me hace falta cargar el modelo y puedo ir viendo con la generacion como van saliendo, aunque es un % muy pequeño ayuda
QUESTIONS_FOR_TESTING =[  
    "In computer vision, what is the RGB (Red, Green, Blue) color space?",
    "what is the main idea behind Otsu’s thresholding method?",
    "What advantages do 3×3 matrices provide?",
    "In computer vision, what is the effect of blurring on image details?",
    "Why is Harris not scale-invariant in computer vision?",
    "What is the idea behind Random Sample Consensus (RANSAC) in computer vision?",
    "How does median filtering work to remove salt-and-pepper noise in computer vision?",  
]

if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    train_config.gradient_accumulation_steps //= ddp_world_size

else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = (
    train_config.gradient_accumulation_steps
    * train_config.batch_size
    * GPTConfig.block_size
    * ddp_world_size
)
if master_process:
    print("Tokens per iteration:", tokens_per_iter)
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(42 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# optimización, en teoría multiplica mas rápido
# https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("high")

dtype_map = {"float16": torch.float16,"bfloat16": torch.bfloat16,"float32": torch.float32,}
autocast_dtype = dtype_map.get(train_config.dtype, torch.bfloat16)

ctx =(
    torch.autocast(device_type="cuda", dtype=autocast_dtype)
    if "cuda" in train_config.device and torch.cuda.is_available()
    else nullcontext()
)


model_args = dict(
    n_layer=GPTConfig.n_layer,
    n_head=GPTConfig.n_head,
    n_embed=GPTConfig.n_embed,
    block_size=GPTConfig.block_size,
    bias=GPTConfig.bias,
    vocab_size=GPTConfig.vocab_size,
    dropout=GPTConfig.dropout,
)

iter_batches = partial(
    Task.iter_batches,
    batch_size=train_config.batch_size,
    max_seq_len=GPTConfig.block_size,
    device=train_config.device,
    num_workers=0,
    data_mode=train_config.data_mode, # modo de entrenamiento text/qa
    tokenizer=tokenizer,
    loss_mode=train_config.loss_mode, # tipo de calcula de la perdida all/answer only
)

best_val_loss = 1e9
# añado una para guardar la mejor de ese run? TODO : ver si me hace falta ahora que guardo cada x iters
best_val_loss_run = 1e9

if resume:
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=train_config.device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf).to(train_config.device)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    best_val_loss = checkpoint["best_val_loss"]

    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    if master_process:
        print("Loaded checkpoint")
else:
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(train_config.device)

scaler = torch.GradScaler(enabled=(train_config.dtype == "float16"))
optimizer = model.configure_optimizers(
    train_config.weight_decay,
    train_config.learning_rate,
    (train_config.beta1, train_config.beta2),
    train_config.device,
)

if train_config.compile:
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    
raw_model = model.module if ddp else model


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(train_config.eval_iters)
        batch_iter = iter_batches(split=split)
        for k in range(train_config.eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                _, loss = raw_model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


@torch.no_grad() 
def save_middle_models(iter_num):
    s = Path(out_dir) / "eval_generations_folder"
    s.mkdir(parents=True, exist_ok=True)

    # Añado
    s_patch = s / f"iter_n_{iter_num}.txt"

    raw_model.eval()  
    with open(s_patch, "w", encoding="utf-8") as f:
        f.write(f"[ITER {iter_num}]\n\n")
       
       # entramos en el bucle de preguntas
        for question in QUESTIONS_FOR_TESTING:
            idxs = tokenizer.encode(question, bos=True, eos=False)
            x = torch.tensor(idxs, dtype=torch.long, device=train_config.device).unsqueeze(0)

            # guardamos la longitud del prompt
            prom_length = x.size(1)

            #Generemos las preguntas/respuestas
            with ctx:  
                y = raw_model.generate(x,
                    max_new_tokens=256, # mismos valores que en el sample
                    temperature=0.1, 
                    top_k=1, # nos quedamos solo con 1
                    top_p=None, 
                    min_p=0.0,  
                )
 
            # ignoramos la pregunta nos quedamos solo la repsuesta
            answer_ids = y[0, prom_length:].tolist()
            # decodificamos y generamos la repsuesta
            answer = tokenizer.decode(answer_ids).replace("\\n", "\n").strip()
            # le doy un poco de formato para que se lea mejor
            f.write(f"Question-> {question}\n")   
            f.write(f"Answer-> {answer}\n") 
            f.write("\n" + "-----------------------------------" + "\n\n")
  
    raw_model.train()
    return s_patch

def get_lr(it):
    if it < train_config.warmup_iters:
        return train_config.learning_rate * (it + 1) / (train_config.warmup_iters + 1)
    if it > train_config.lr_decay_iters:
        return train_config.min_lr
    decay_ratio = (it - train_config.warmup_iters) / (
        train_config.lr_decay_iters - train_config.warmup_iters
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_config.min_lr + coeff * (
        train_config.learning_rate - train_config.min_lr
    )


params = sum([p.numel() for p in model.parameters() if p.requires_grad])

if master_process:
    print("🖥️==Starting training==🖥️\n")
    print("The number of params is :", params)

train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)

iter_num = 0
t0 = time.time()

while True:
    lr = get_lr(iter_num) if train_config.decay_lr else train_config.learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % train_config.eval_interval == 0 and master_process:
        losses = estimate_loss()
        # 
        print(f"\n=== EVAL {iter_num} ===")
        print(f"train_loss : {losses['train']:.4f}")
        print(f"val_loss   : {losses['val']:.4f}")
        print(f"best_run   : {best_val_loss_run:.4f}")
        print(f"best_global: {best_val_loss:.4f}")

        writer.add_scalar("train_loss", losses["train"], iter_num)
        writer.add_scalar("val_loss", losses["val"], iter_num)
        writer.add_scalar("lr", lr, iter_num)

        # Mejor de esta run TODO ESTO DEBO QUEDARMELO? ES EL MEJOR DE LA RUN
        if losses["val"] < best_val_loss_run:
            best_val_loss_run = losses["val"]
            checkpoint_run_best = {
                "model": raw_model.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss_run,
            }
            torch.save(checkpoint_run_best, os.path.join(out_dir, "ckpt_best_run.pt"))
            print("✅ New best model saved in -> ckpt_best_run.pt")
        else:
            print("ℹ️ ckpt_best_run.pt has no improvements in this iteration")

        # Mejor histórico/global
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            checkpoint = {
                "model": raw_model.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
            print("✅ Nuevo mejor histórico -> ckpt.pt")
        else:
            print("ℹ️ ckpt.pt no mejora en esta evaluación")

        s_patch = save_middle_models(iter_num)
        print(f"📄 Generaciones guardadas en: {s_patch}")
        
        # aqui hago un if para guardar el modleo cada x iteraciones, en fases tempranas me quedo con varios modelos, se peude comentar este bloque, en este caso paso de las primeras  y a partir de la 4000 me qguardo un modelo
        if iter_num % 1000 == 0 and iter_num > 3000:
        
            checkpoint_iter = {
                "model": raw_model.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
            }
            # guardamos 
            torch.save(checkpoint_iter,os.path.join(out_dir, f"ckpt_iter_{iter_num}.pt"))
            print(f"✅Modelo intermedio guardado en -> ckpt_iter_{iter_num}.pt")
            
        # fin del bloque para guardar modelos cada x iters              
        print("=== FIN EVALUATION ===\n")
    for micro_step in range(train_config.gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (
                micro_step == train_config.gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / train_config.gradient_accumulation_steps

        X, Y = next(train_batch_iter)
        scaler.scale(loss).backward()

    if train_config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    iter_num += 1

    # Guardo una copia al final del ultimo train #TODO MIRAR SI QUITARLO YA QUE GUARDO X ITERS
    if iter_num > train_config.max_iters:
        if master_process:
            checkpoint_last = {
                "model": raw_model.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint_last, os.path.join(out_dir, "ckpt_last.pt"))
            print("💾 Last checkpoint saved  -> ckpt_last.pt")
        break

if ddp:
    destroy_process_group()
writer.close()

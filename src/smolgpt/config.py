from dataclasses import dataclass
import torch


@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 4096
    n_layer: int = 8
    n_head: int = 8
    n_embed: int = 512
    dropout: float = 0.00
    bias: bool = False
    use_rotary: bool = False


#10.000 iters ejemplo
@dataclass
class TrainingConfig:
    learning_rate: float = 6e-4
    max_iters: int = 10000

    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    decay_lr: bool = True
    warmup_iters: int = 400   
    lr_decay_iters: int = 10000
    min_lr: float = 6e-5

    eval_interval: int = 500  
    log_interval: int = 50
    eval_iters: int = 30

    gradient_accumulation_steps: int = 2
    batch_size: int = 64

    device: str = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dtype: str = "float16"
    compile: bool = False


    # Definimos dos parámetros adicionales
    data_mode : str = "qa" # text o qa , text es el entrenamiento que usa tinystories y smolgpt, qa es el formato de preguntas y repsuestas como se ve en el corpus del trabajo
    loss_mode : str = "answer_only" # answer_only/full , si la loss se calcula solo en la respuesta o en pregunta-respuesta
    
    
# Configuración original de tinystories
@dataclass
class TrainingConfigOriginal:
    learning_rate: float = 6e-4
    max_iters: int = 30000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    decay_lr: bool = True
    warmup_iters: int = 1000
    lr_decay_iters: int = 30000
    min_lr: float = 6e-5

    eval_interval: int = 100
    log_interval: int = 10
    eval_iters: int = 200
    gradient_accumulation_steps: int = 4
    batch_size: int = 64

    device: str = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dtype: str = "bfloat16"
    compile: bool = True

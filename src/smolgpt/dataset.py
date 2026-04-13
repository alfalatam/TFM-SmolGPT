import torch
import numpy as np
import random
from pathlib import Path
import glob
from typing import Iterator, Tuple
import torch.distributed as dist
import os

# Dataset para cuando el corpus esta compuesto por texto como libros
class PreTokDataset(torch.utils.data.IterableDataset):
    def __init__(self, split: str, max_seq_len: int):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = 42 + worker_id + 1337 * rank
        #bin_dir = Path("data/TinyStories_all_data")
        bin_dir = Path("data")

        shard_filenames = sorted(glob.glob(str(bin_dir / "*.bin")))
        shard_filenames = (
            shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        )
        rng = random.Random(seed)
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                data = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(data) // self.max_seq_len - 1
                idxs = list(range(num_batches))
                rng.shuffle(idxs)

                for idx in idxs:
                    start = idx * self.max_seq_len
                    end = (idx + 1) * self.max_seq_len
                    chunk = torch.from_numpy(data[start:end].astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y


# dataset para cuando entrenamos el modleo con un formato de QA
class QADataset(torch.utils.data.IterableDataset):

    # definimos y guardo la config del dataset
    def __init__(self, split: str, tokenizer, max_seq_len: int, loss_mode: str = "full"):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        # modo de calcular la pérdida si solo repuesta o todo full o answer_only que se configuta en config
        self.loss_mode = loss_mode
        self.data_dir = Path("data/libros")

    # como mi txt es linea Q, linea A y espacio en blanco con esta funcion le pongo como bloques separados
    def split_bloqs(self, text: str):
        lines = text.splitlines()
        bloqs = []
        qa = []

        for ln in lines:
            if ln.strip() == "":
                if qa:
                     bloqs.append(qa)
                     qa =[]
            else: 
                qa.append(ln.strip())

        if qa:
            bloqs.append(qa)

        return bloqs

    # funcion de carga
    def load_examples(self):
        examples = []
        # recorro los txt 
        for text in sorted(os.listdir(self.data_dir)):
        # pasamos de los que no sean txt
            if not text.endswith(".txt"):
                continue

            path =self.data_dir / text
            with open(path, "r", encoding="utf-8") as f:  
                text = f.read() 
            # llamsmoa a la funcion anterior para ir generando los bloques
            bloqs = self.split_bloqs(text)

            for b in bloqs:
                if len(b) < 2:
                    continue
                q = b[0]
                a = b[1]
                if len(b) > 2:
                    a = a + " " + " ".join(b[2:])

                examples.append((q.strip(), a.strip()))

        if not examples:
            raise ValueError("No se encontraron ejemplos Q/A en laruta data/libros")

        return examples

    # aqui pytorch va pidiendo los ejemplos para entrenar 
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)

        # cargamos los ejemplos
        examples = self.load_examples()
        
        examples = self.select_split(examples)
        
        while True:
            rng.shuffle(examples)
            
            for q,a in examples:
            
                x,y = self.prepare_one_example(q,a)
                yield x,y



    def prepare_one_example(self, q: str, a:str):
        ids, q_len, origi_length = self.encode_qa(q, a)
        
        ids = self.pad_or_slice(ids)
        x,y= self.generate_xy(ids)
        
        if self.loss_mode == "answer_only":
            y = self.loss_mask(y ,q_len,origi_length)
            
        else:
            y = self.mask_padding_only(y,origi_length)
            
        return x,y
    
    
    # funcion aux para entrenar sobre todo o split
    def select_split(self,examples):
    
        split_dataset =  getattr(self, "split_dataset", False)
        
        if not split_dataset:
            if self.split in ["train","val"]:
                return examples
            raise ValueError(f"SPlit not supported : {self.split}")
        
        split_index = int(len(examples)* 0.9) #  % de corpus para train y val
        
        if self.split == "train":
            return examples[:split_index]
        if self.split == "val":
            return examples[split_index:]
        
        raise ValueError(f"Split not supported : {self.split}")
        
        
        
    def encode_qa(self, q: str, a: str):
    
        question_ids = self.tokenizer.encode(q, bos=True, eos=False)
        nline_id = self.tokenizer.encode("\n", bos=False, eos=False)
        answer_ids = self.tokenizer.encode(a, bos=False, eos=True)
                
        # concatenamos todos los ids generando la solución que tenemos al princpio
        ids = question_ids+nline_id+answer_ids               
        q_len= len(question_ids) +len(nline_id)
        origi_length = min(len(ids), self.max_seq_len)

        return ids,q_len,origi_length
        
        
    
    def pad_or_slice(self, ids):
        pad_token = self.tokenizer.pad_id
        
        if pad_token == -1:
            pad_token = self.tokenizer.eos_id
            
        if len(ids)> self.max_seq_len:
            return ids[:self.max_seq_len]
            
        return ids + [pad_token] * (self.max_seq_len - len(ids))
        
    
    def generate_xy(self,ids):
        x= torch.tensor(ids[:-1], dtype=torch.long)
        y= torch.tensor(ids[1:],dtype = torch.long)
        
        return x,y


        
    def loss_mask(self, y ,q_len,origi_length):
        y_mask = y.clone()
        ignored = max (0, q_len-1)
        y_mask[:ignored] = -1
                    
        # ignoramos el padding final, desde el final de la secuencia hasta el final
        if origi_length < self.max_seq_len:
            y_mask[origi_length -1:] = -1
                    
                    
        return y_mask
        
        
    def mask_padding_only(self,y,origi_length):
        if origi_length < self.max_seq_len:
                    y_mask = y.clone()
                    y_mask[origi_length - 1:] = -1
                    return y_mask
        return y

            
    


                
# devolvemos los batches x,y preparados para el proyecto preparando los datos en buble
class Task:
    @staticmethod
    def iter_batches(
        batch_size: int,device: str,num_workers: int = 0,split: str = "train",max_seq_len: int = 512,
        data_mode: str = "text",tokenizer=None,loss_mode: str = "full",
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:

        if data_mode == "qa":
            ds = QADataset(split=split,tokenizer=tokenizer,max_seq_len=max_seq_len,loss_mode = loss_mode,)
        else:
            ds = PreTokDataset(split=split,max_seq_len=max_seq_len,)

        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, num_workers=num_workers
        )
        # se lo pasamos al device (gpu o cpu)
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            yield x, y

# =====================


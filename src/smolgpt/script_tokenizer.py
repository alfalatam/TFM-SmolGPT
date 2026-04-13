import os
import numpy as np
from tqdm import tqdm
from smolgpt.tokenizer import Tokenizer

#  directorios
TOKENIZER_MODEL_PATH = "data/tok4096.model"
LIBROS_DIR = "data/libros"
TRAIN_FILE = "data/1_train.bin"
VAL_FILE = "data/0_val.bin"


# funcion para dividir el texto en bloques siendo cada uno una pregunta
def split_blocks(text: str):
    lines = text.splitlines()
    blocks, x = [], []
    # recorremos el archivo
    for line in lines:
        # recorre el texto si encuentra linea vacia, es un bloque , guardamos y reiniciamos
        if line.strip() =="":
            if x:
                blocks.append(x)
                x = []
        else:
            x.append(line.strip() )
    if x:
       blocks.append(x)
    return blocks


# funcion de preprocesamiento usando los txt realizamos la division en un ratio 90-10
def preprocesamiento(libros_dir, tokenizer_model, train_file, val_file, split_ratio=0.9):
    
    # inicialzamos el model
    tokenizer = Tokenizer(tokenizer_model)
    id_list = []
    # recorremos los libros sin son txt para exitar errores
    for libro in tqdm(os.listdir(libros_dir), desc=" ⚙️====Tokenizando libros====⚙️"):
        if not libro.endswith(".txt"):
            continue

        path = os.path.join(libros_dir, libro)
        # cargamos el contenido del libro
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        # llamamos a la funcion split para separar 
        blocks = split_blocks(text)
        
        # aqui recorro los bloques siendo un bloque una QA
        for b in blocks:
            # una comprobacion simple por si algun bloque est mal hecho
            if len(b) < 2:
                continue
            # guardamos la pregunt ay respuesta por separado
            q = b[0]
            a = b[1]
            # si hay mas lineas los unimos
            if len(b) > 2:
                a = a + " " + " ".join(b[2:])
            # creamos lo que vamos a guardar en el rxr
            sample = q.strip() + "\n" + a.strip()

            # BOS/EOS por ejemplo (por cada par Q/A)
            # guardamos los tokens
            ids = tokenizer.encode(sample, bos=True, eos=True)
            id_list.extend(ids)

    arr = np.array(id_list, dtype=np.uint16)

    # hacemos la division segun el ratio y guardamos los archivos
    split_idx = int(len(arr) * split_ratio)
    train_data = arr[:split_idx]
    val_data = arr[split_idx:]
    train_data.tofile(train_file)
    val_data.tofile(val_file)

    print("\n")
    print("\n")
    print(f"Se ha generado {len(arr)} tokens en total")
    print(f"Se ha destinado para el entrenamiento: {len(train_data)} tokens")
    print(f"Se ha destinado para la validación: {len(val_data)} tokens")
    print("\n")
    print("\n")



# definimos el preprocesamiento dentro de main paara llamarlo desde tools
#TODO: Mirar porque me sale como que se procesan 2/2 cuando solo tengo un archivo
def main():
    print("Preparando datos de los libros.....")
    preprocesamiento(LIBROS_DIR, TOKENIZER_MODEL_PATH, TRAIN_FILE, VAL_FILE)
    print("\n ====Archivos .bin generados en la carpeta data====")
    print(" ====Todo preparado para el entrenamiento====\n")

# 
if __name__ == "__main__":
    preprocesamiento(LIBROS_DIR, TOKENIZER_MODEL_PATH, TRAIN_FILE, VAL_FILE)
    main()

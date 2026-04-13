import numpy as np

# Función auxiliar para calcular el numero de tokens
def main():
# poner el .bin que se quiera comprobar
    tokens = np.fromfile("data/1_train.bin", dtype=np.uint16)
    print("Number of tokens:", len(tokens))
    print("Unique tokens:", len(set(tokens)) )
  
  
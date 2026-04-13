import numpy as np
from smolgpt.tokenizer import Tokenizer

# Función auxiliar que muestra el numero de veces que aparece los tokens de inicio y fin de secuencia (BOS y EOS) para un bin ,asi compruebo si se estan generando bien las QA

def main():

    # cargamos el tokenizer
    tokenizer = Tokenizer("data/tok4096.model")
    # Cargamos el .bin
    bin_path = "data/1_train.bin"       
    data = np.fromfile(bin_path, dtype=np.uint16)
    t_tokens = len(data)
    
    

    def contar(token_id, nombre):

        contador = np.sum(data == token_id)
        freq = contador / t_tokens
        print(f"{nombre}:")
        print(f"-ID: {token_id}")
        print(f"-Number of times: {contador}")
        print(f"-Frecuency: {freq:.3f}")
        print("=============================================================" )


    print("=========== Special IDS detected ================")
    print(f"El token <s>(BOS) tiene el id: {tokenizer.bos_id}")
    print(f"El token </s>(EOS) tiene el id: {tokenizer.eos_id}")
    
    print("=============================================================" )
    print(f"El número total de tokens de los que se dispone es: {t_tokens}")
    print("=============================================================" )
    # calculamos el numero y frecuencia de aparición del EOS y BOS
    contar(tokenizer.bos_id, "<s> (BOS) ")
    contar(tokenizer.eos_id, "</s> (EOS) ")

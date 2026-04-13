from smolgpt.tokenizer import Tokenizer



def main():
    # Ruta a tu modelo de tokenización
    t_path = "data/tok4096.model"
   
    t = Tokenizer(t_path)

    # Archivo de salida
    output_file = "data/vocabulary/vocabulary.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(t.sp_model.vocab_size()): 
            token = t.sp_model.id_to_piece(i) 
            f.write(f"{i}\t{token}\n")
            
            
            
    print(f"Vocabulary exported as a txt file in : {output_file}")
    print(f"Total de tokens: {t.sp_model.vocab_size()}")
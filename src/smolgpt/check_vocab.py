from smolgpt.tokenizer import Tokenizer

def main():

    t = Tokenizer("data/tok4096.model")

    print("Hello, this function allows you to check whether a word is in the vocabulary and how it is formed.")

    while True:
    
        text = input("\nEnter a sentence or a word (exit or x to leave this mode): ")
        
        #salimos del bucle si escribimos.
        if text.lower() == "exit" or text.lower() == "x":
            break

        print("\n-- The tokenization is --")
        
        # recorremos los tokens
        listaTokens = t.encode(text, bos=False, eos=False)
        for token in listaTokens:
            piece = t.sp_model.id_to_piece(token)
            print(f"The ID is{token}and the token is {piece}")
            
           

        print("\n--- This checks whether the word is in the vocabulary ---")
        
        vocab= []
       # para cada id, obtenemos el token y lo añadimso a la lista
        for i in range(t.sp_model.get_piece_size()):
            piece = t.sp_model.id_to_piece(i)
            vocab.append(piece)
        # ▁ es un caracter de setence pice para marar el inicio de una palabra
        if text in vocab or ("▁" + text in vocab):
            print("The word DOES exist as an complete token in the vocabulary.")
        else:
            print("The word DOES NOT exist as an complete token in the vocabulary.")
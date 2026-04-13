from smolgpt.tokenizer import Tokenizer

# funcion auxiliar para ver como queda una palabra o frase al tokenizarla
def main():
    # cargar tokenizer
    t = Tokenizer("data/tok4096.model")

    print("Tokens counter")
    print("Write and check the tokenization.")
    print("Press 'exit' to leave.\n")

    # nos metemos en un bucle
    while True:
        text = input("Text: ")

        # salimos del bucle
        if text.lower() in ["exit", "salir", "x"]:
            break

        tokens = t.encode(text, bos=False, eos=False)
        print("Number of tokens: ", len(tokens))
        print("IDs:", tokens)
        print("------------------------------------------")
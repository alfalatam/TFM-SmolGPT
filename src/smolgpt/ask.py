from smolgpt.tokenizer import Tokenizer
from smolgpt.sample import load_model, setup_device
import torch
import argparse

# Parámetros
MAX_TOKENS= 256 # el mismo que he usado en el train
DEVICE= "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # Reutilizamos load_model de sample.py — necesita un args con los campos que usa
    args = argparse.Namespace(
        ckpt_path="out/ckpt_best_run.pt",
        device= DEVICE,        # usamos gpu si podemos, sino vamos por cpu
        dtype="float16",compile=False,seed=42,)
        
    # cargamos el modelo.
    tokenizer = Tokenizer("data/tok4096.model")
    ctx       = setup_device(args)
    model     = load_model(args)
    
    print("\nHi! I'm a virtual assistant, and I'm here to help you understand computer vision concepts. Ask me any questions, and I'll do my best to help you..")
    print("Please enter your question (if you want to exit, type “x”, ‘exit’,, ‘quit’ or “salir”):\n")
    
    while True:
        question = input("Question: ").strip()
        
       # compruebo si ha escrito algo o si quiere terminar con las preguntas
        if not question:
            continue
        if question.lower() in ["x","exit","quit", "salir"]:
            print("Bye, Have a nice day!")
            
            break

        # Mantenemos el mismo formato que el corpus, para obtener respuestas parecidas que en el entrenamiento
        prompt = f"{question}\n"
        # tokenizo la pregunta y lo convertimos en un tensor
        ids    = tokenizer.encode(prompt, bos=True, eos=False)
        x      = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        
        # como es inferencia.
        with torch.no_grad():  
            with ctx: 
                # los params son los mismos que he usado en el train
                out = model.generate(x, MAX_TOKENS, temperature=0.1, top_k=1)

        answer_ids = out[0, len(ids):].tolist()
        if tokenizer.eos_id in answer_ids:
            answer_ids = answer_ids[:answer_ids.index(tokenizer.eos_id)]

        
        answer = tokenizer.decode(answer_ids).replace("\\n", "\n").strip()

        print(f"\nAnswer: {answer}\n")
        print("--------------------------------------------------------------")
        
 

        
if __name__ == "__main__":
    main()
# importamos
import torch
import json
import torch.nn.functional as F
from smolgpt.tokenizer import Tokenizer
from smolgpt.sample import load_model,setup_device
from types import SimpleNamespace

# Path
PATH = "tests/test.json"

# config y carga dle modelo
def build_args():
    
    return SimpleNamespace(
        ckpt_path="out/ckpt_best_run.pt",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="float16", 
        compile=False,
        seed=42,
    )
    
def load_test():
    with open(PATH,"r", encoding = "utf-8") as f:
       return json.load(f)


# calculo la log prob  y luego evaluo la pregunta

def score(model,tokenizer,ctx,device,question,candidate):

    prompt_question_ids = tokenizer.encode(question.strip() + "\n", bos = True,eos = False)
    answer_test_ids = tokenizer.encode(candidate.strip(), bos = False, eos = True)
    
    ids = prompt_question_ids + answer_test_ids
    # x lo que ve el modelo y Y es la prediccion
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        with ctx:
            logits, _ = model(x, y)
            
    
    logs_prob = F.log_softmax(logits,dim=-1)
    start = len(prompt_question_ids) - 1 # -1 porque x/y están deplsazados uno
    scores = []
    
    for position in range(start, start+len(answer_test_ids)):
        target_token = y[0,position].item()
        token_logs_prob= logs_prob[0,position,target_token].item()
        scores.append(token_logs_prob)
    
    # calculamos la media de las l.prob  de la respuesta , si no hay tokens devolvemos menos infinito
    if len(scores)>0:
        return sum(scores)/len(scores)
    else:
        return float("-inf")
    


def eval_one_question(model,tokenizer,ctx,device,item):

    scores ={opcion:score(model,tokenizer,ctx,device,item["question"],item[opcion])
    for opcion in ["A", "B", "C", "D"] }
    
    orderScore = sorted(scores, key = scores.get, reverse = True)
    
    top1 = orderScore[0]
    top2 = orderScore[:2]
    margin= scores[top2[0]] - scores[top2[1]]
    
    return top1,top2,margin,scores
    
    
def main():
    
    args = build_args()
    tok = Tokenizer("data/tok4096.model")
    ctx = setup_device(args)
    model =load_model(args)
    questions = load_test()
    top1,top2= 0,0
    
    print("\n ========== STARTING TEST===============")
    for question in questions:
        pred,best2,margin,scores = eval_one_question(
        model,tok,ctx,args.device,question)
        
        
        
        
        corr_answer = question["answer"].strip().upper()
        
        top1 += pred == corr_answer
        top2 += corr_answer in best2
        
        # aqui mostramos por pantalla los resultados.
        
        print(f"Q: {question['question']}")
        print(f"Correct answer: {corr_answer}")
        print(f"Prediction: {pred} || Top2 Prediction:{best2}|| Margin: {margin:.3f}")
        print("Scores:", {key: f"{value:.3f}" for key, value in scores.items()}) 
        print("\n")
    total = len(questions)
    
    # calculamos la nota tanto de top1 como top2
    
    print(f"Top1 Test accurracy: {top1/total:.2%}")
    print(f"Top2 Test accurracy: {top2/total:.2%}")
    if(top1/total >= 0.5):
        print("===================")
        print("==== TEST PASSED ====")
        print("===================")

    else:
        print("===================")
        print("==== TEST FAILED ====")
        print("===================")


if __name__ == "__main__":
    main()
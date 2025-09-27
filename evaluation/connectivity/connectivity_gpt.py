import openai
import json
import os
from tqdm import tqdm
import networkx as nx
import numpy as np
import argparse
import time
from datetime import datetime, timedelta, timezone
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from openai import OpenAI
import ipdb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

model_list = ["gpt-3.5-turbo","o4-mini","gpt-4.1-mini","gpt-4o"]
parser = argparse.ArgumentParser(description="shortest path")
parser.add_argument('--model', type=str, default="text-davinci-003", help='name of LM (default: text-davinci-003)')
parser.add_argument('--prompt', type=str, default="none", help='prompting techniques (default: none)')
parser.add_argument('--T', type=int, default=0, help='temprature (default: 0)')
parser.add_argument('--token', type=int, default=4000, help='max token (default: 1000)')
parser.add_argument('--SC', type=int, default=0, help='self-consistency (default: 0)')
parser.add_argument('--city', type=int, default=0, help='whether to use city (default: 0)')
parser.add_argument('--SC_num', type=int, default=5, help='number of cases for self-consistency (default: 5)')
parser.add_argument('--begin', type=int, default=0, help='beginning')
parser.add_argument('--end', type=int, default=100, help='ending')
parser.add_argument('--source_path', type=str, default="json", help='ending')
args = parser.parse_args()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def translate(G, q, args):
    Q = ''
    if args.prompt in ["CoT", "k-shot","Algorithm","Instruct"]:
        with open("NLGraph/connectivity/prompt/" + args.prompt + "-prompt.txt", "r") as f:
            exemplar = f.read()
        Q = Q + exemplar + "\n\n\n"
    
    Q = Q + G

    Q = Q + "Q: Is there a path between node " + str(q[0])+" and node " + str(q[1]) + "?\n"
    Q = Q + "Please only answer with 'Yes' or 'No'."

    return Q


@retry(wait=wait_random_exponential(min=1, max=600), stop=stop_after_attempt(10))
def predict(Q, args):
    input = Q
    temperature = 0
    if args.SC == 1:
        temperature = 0.7

    Answer_list = []
    for text in input:
        #print(f"Sending request to DeepSeek API with text: {text}")
        #print("\n")
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text},
            ],
            #temperature=temperature,
            max_completion_tokens=args.token,
            stream=False,
            timeout=100  
        )
        Answer_list.append(response.choices[0].message.content.strip())
        print(f"response:{response.choices[0].message.content}")

    return Answer_list


def evaluate(ans,std):
    if "yes" in ans :
        v1=1
    elif "no" in ans:
        v1=0
    else:
        v1 = -1
    if v1 == -1:
        return  0, 0
    if str(std) == "Yes":
        v2 = 1
    elif str(std) == "No":
        v2 = 0
    else:
        return 0, 0
    flag1=1
    if v1 == v2:
        flag2 = 1
    else:
        flag2 = 0

    print("GT answer: ", std)
    return flag1, flag2

def main():
    if 'OPENAI_API_KEY' in os.environ:
        openai.api_key = os.environ['OPENAI_API_KEY']
    else:
        raise Exception("Missing openai key!")
    if 'OPENAI_ORGANIZATION' in os.environ:
        openai.organization = os.environ['OPENAI_ORGANIZATION']
    res1,  res2, answer, answer_all = [], [], [], []

    g_num =  args.end-args.begin #200

    batch_num = 1

    file_path = 'data/connectivity.json'
    start_idx = args.begin-1
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)  
    for i in tqdm(range((g_num + batch_num - 1) // batch_num)):
        G_list, Q_list, q_list, a_list= [], [], [], []
        start_idx +=1
        for j in range(start_idx, start_idx+1):
            entry = data[j]        
            if 'query_clustersplit' in entry:
                G= entry['query_clustersplit']  
            if 'aim' in entry:
                q = entry['aim']
            if 'result' in entry:
                a = entry['result']


            Q = translate(G, q, args)
            Q_list.append(Q)
            G_list.append(G)
            q_list.append(q)
            a_list.append(a)
            if j==0:
                print("/n")
            print(f"Processing batch {i + 1}, graph {j}")

        sc = 1
        if args.SC == 1:
            sc = args.SC_num
        sc_list = []
        for k in range(sc):
            answer_list = predict(Q_list, args)
            answer_all.append(answer_list) 
            sc_list.append(answer_list)
        for j in range(len(Q_list)):
            vote1, vote2 = 0, 0
            for k in range(sc):
                ans, G = sc_list[k][j].lower(), G_list[j]
                answer.append(ans.lower())
                try:
                    r1, r2 = evaluate(ans.lower(), a_list[j]) 
                    vote1 += r1
                    vote2 += r2
                except:
                    print(ans.lower())
            r1 = 1 if vote1*2 > sc else 0
            r2 = 1 if vote2*2 > sc else 0 
            res1.append(r1)
            res2.append(r2)

    y_true = []
    y_pred = []

    for i, entry in enumerate(data[args.begin : args.end]):
        true_label = 1 if entry['result'] == 'Yes' else 0
        y_true.append(true_label)

        if res2[i] == 1:
            y_pred.append(true_label) 
        else:
            y_pred.append(1 - true_label)  


    y_true = np.array(y_true)
    y_pred = np.array(y_pred)


    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(len(y_true), len(y_pred))

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=['No', 'Yes']))
    merged = list(zip(res1, res2))
    res1 = np.array(res1)
    res2 = np.array(res2)
    answer = np.array(answer)
    print("res2 for answer right",res2.sum()) 
    print("res1 for Extraction successfullu",res1.sum()) 
    print("Answer: ", merged)

if __name__ == "__main__":
    main()

import openai
import json
import os, re
from tqdm import tqdm
import numpy as np
import argparse
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from openai import OpenAI

model_list = ["gpt-3.5-turbo","o4-mini","gpt-4.1-mini","gpt-4o","gpt-5.1","gpt-5.2","gpt-5","gpt-5-mini","gpt-o3"]
parser = argparse.ArgumentParser(description="shortest path")
parser.add_argument('--model', type=str, default="text-davinci-003", help='name of LM (default: text-davinci-003)')
parser.add_argument('--mode', type=str, default="easy", help='mode (default: easy)')
parser.add_argument('--prompt', type=str, default="none", help='prompting techniques (default: none)')
parser.add_argument('--T', type=int, default=0, help='temperature (default: 0)')
parser.add_argument('--token', type=int, default=8192, help='max token (default: 1000)')
parser.add_argument('--begin', type=int, default=0, help='beginning')
parser.add_argument('--end', type=int, default=100, help='ending')
args = parser.parse_args()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def translate(G, args, number, node, edge, flag):
    Q = ''
    with open("newdata/Graph-Reasoning-LLM-main/dataset/triangle.txt", "r") as f:
        exemplar = f.read()
    Q = Q + exemplar + "\n\n\n"

    if flag == 1:
        Q = Q + "The current graph is divided into two clusters.\n"

    Q = Q + "The maximum sum of the weights of three nodes in cluster0 and the direct connectivity of the exit nodes in its cluster0 are " + '\n'

    file_path = "traindata/tri/i_0/graph" + str(number) + "_0.txt"
    
    with open(file_path, "r", encoding="utf-8") as f:
        exemplar = f.read()
        Q = Q + exemplar + "\n"

    Q = Q + "The maximum sum of the weights of three nodes in cluster1 and the direct connectivity of the exit nodes in its cluster1 are " + '\n'

    file_path = "traindata/tri/i_1/graph" + str(number) + "_1.txt"
    
    with open(file_path, "r", encoding="utf-8") as f:
        exemplar = f.read()
        Q = Q + exemplar + "\n"
    
    if flag == 1:
        Q = Q + "The edges that directly connect the two clusters are: "
    Q = Q + edge + ' '
    Q = Q + "All exit nodes of the two clusters are: "
    Q = Q + node + '\n'
    Q = Q + "Q: What is the maximum sum of the weights of three nodes? "

    return Q

@retry(wait=wait_random_exponential(min=1, max=600), stop=stop_after_attempt(10))
def predict(Q, args):
    temperature = 0

    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": Q},
        ],
        temperature=temperature,
        max_completion_tokens=args.token,
        stream=False,
        timeout=200
    )
    answer = response.choices[0].message.content.strip()
    print(f"response:{answer[:400] if len(answer) > 400 else answer}")
    return answer

def evaluate(ans, a):
    print("max GT is: ", a)
    start = ans.lower().find("final answer")
    if start == -1:
        return 0, 0
    text = ans[start:]

    key = "the maximum sum of the weights"
    pos = text.find(key)
    if pos == -1:
        return 0, 0

    match = re.search(r'\d+', text[pos + len(key):])
    if not match:
        return 0, 0
    num = int(match.group())

    print("gpt answer: ", num)
    print("max: ", a)

    return (1, 1) if num == a else (1, 0)

def main():
    if 'OPENAI_API_KEY' not in os.environ:
        raise Exception("Missing openai key!")
    
    res1, res2 = [], []

    match args.mode:
        case "easy":
            g_num = 180
        case "hard":
            g_num = args.end - args.begin  # 200

    start_idx = args.begin
    file_path = ''
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)  

    res1, res2 = [], []
    for i in tqdm(range(g_num)):
        idx = start_idx + i
        if idx >= len(data):
            break

        entry = data[idx]

        if 'result' in entry:
            a = entry['result']
        if 'Cross_edges' in entry:
            edge = entry['Cross_edges']
        if 'All_exit_nodes' in entry:
            nodes = entry['All_exit_nodes']
        
        G = None
        flag = 1
        
        Q = translate(G, args, idx, nodes, edge, flag)
        
        print(f"\nProcessing graph {idx}...")

        answer_text = predict(Q, args)
        
        r1, r2 = evaluate(answer_text.lower(), a)
        res1.append(r1)
        res2.append(r2)
            
            
    
    
    
    # 显示评估结果
    if res1:
        res1 = np.array(res1)
        res2 = np.array(res2)
        print(f"\nTotal graphs processed: {len(res1)}")
        print("res2 for answer right", res2.sum())
        print("res1 for extraction successful", res1.sum())


if __name__ == "__main__":
    main()
import openai
import json
import os
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
parser.add_argument('--mode',type=str, default="easy", help='mode (default: easy)')
parser.add_argument('--prompt', type=str, default="none", help='prompting techniques (default: none)')
parser.add_argument('--T', type=int, default=0, help='temperature (default: 0)')
parser.add_argument('--token', type=int, default=8192, help='max token (default: 1000)')
parser.add_argument('--begin', type=int, default=0, help='beginning')
parser.add_argument('--end', type=int, default=100, help='ending')
args = parser.parse_args()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def translate(G, ques, node, exit, edge):
    Q = ''
    with open("newdata/Graph-Reasoning-LLM-main/dataset/triangle-2.txt", "r") as f:
        exemplar = f.read()
    Q = Q + exemplar + "\n\n\n"

    Q = Q + ques
    Q = Q + "The nodes in cluster 0 are: " + node + ". "
    Q = Q + "The edges in cluster 0 are: " + edge + ". "
    Q = Q + "The exit nodes in cluster 0 are: " + exit + ". "
    Q = Q + "Q: What is the maximum sum of the weights of three nodes ? And list all directly connected exit node pairs." + "Answer it started with 'Final Answer:'."

    return Q

@retry(wait=wait_random_exponential(min=20, max=600), stop=stop_after_attempt(10))
def predict(Q, args):
    temperature = 0

    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": Q},
        ],
        temperature=temperature,
        max_completion_tokens=args.token - len(Q),
        stream=False,
        timeout=100
    )
    answer = response.choices[0].message.content
    print(f"response:{answer}")
    return answer

def extract_cluster_info(text, j, args):  
    newpath = 'traindata/tri/i_0/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    final_answer_start = text.find("final answer:")
    if final_answer_start != -1:
        text = text[final_answer_start:]
    else:
        final_answer_start = text.find("final answer")
        if final_answer_start != -1:
            text = text[final_answer_start:]

    with open(newpath + "graph"+str(j)+"_0.txt", "w", encoding="utf-8") as f:
        f.write(text)

def main():
    if 'OPENAI_API_KEY' not in os.environ:
        raise Exception("Missing openai key!")
    
    match args.mode:
        case "easy":
            g_num = 180
        case "hard":
            g_num = args.end - args.begin  # 200

    start_idx = args.begin
    file_path = ''
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)  
    

    for i in tqdm(range(g_num)):
        idx = start_idx + i
        if idx >= len(data):
            break

        entry = data[idx]
        
        if 'ques' in entry:
            ques = entry['ques']
        if 'Cluster_0' in entry:
            node = entry['Cluster_0']
        if 'Exit_nodes0' in entry:
            exit_nodes = entry['Exit_nodes0']
        if 'Intra_0_edges' in entry:
            edges = entry['Intra_0_edges']
        
        G = None
        Q = translate(G, ques, node, exit_nodes, edges)
        
        print(f"\nProcessing graph {idx}...")

        answer_text = predict(Q, args)
        
        extract_cluster_info(answer_text.lower(), idx, args)
            
    


if __name__ == "__main__":
    main()
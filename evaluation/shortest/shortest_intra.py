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

model_list = ["gpt-3.5-turbo","o4-mini","gpt-4.1-mini","gpt-4o"]
parser = argparse.ArgumentParser(description="shortest path")
parser.add_argument('--model', type=str, default="text-davinci-003", help='name of LM (default: text-davinci-003)')
parser.add_argument('--prompt', type=str, default="none", help='prompting techniques (default: none)')
parser.add_argument('--T', type=int, default=0, help='temprature (default: 0)')
parser.add_argument('--token', type=int, default=8192, help='max token (default: 1000)')
parser.add_argument('--SC', type=int, default=0, help='self-consistency (default: 0)')
parser.add_argument('--city', type=int, default=0, help='whether to use city (default: 0)')
parser.add_argument('--SC_num', type=int, default=5, help='number of cases for self-consistency (default: 5)')
parser.add_argument('--split_path', type=str, default="None", help='path is must')
parser.add_argument('--save_path', type=str, default="None", help='path is must')
parser.add_argument('--begin', type=int, default=0, help='beginning')
parser.add_argument('--end', type=int, default=100, help='ending')
parser.add_argument('--input', type=str, default="json", help='json')
parser.add_argument('--number',type=str, default="0", help='number of cluster')
args = parser.parse_args()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def translate(G, q, args, number,ques,node,exit,edge,i):
    # edge = list(G.edges())
    # n, m = G.number_of_nodes(), G.number_of_edges()
    Q = ''
    prompt_folder = "prompt"
    if args.prompt in ["CoT", "k-shot","Algorithm","Instruct",'dot1','dot2','ins1','ins2','ins3']:
        with open("NLGraph/shortest_path/"+prompt_folder+"/" + args.prompt + "-prompt.txt", "r") as f:
            exemplar = f.read()
        Q = Q + exemplar + "\n\n\n"
    else: 
        xx=1

    Q = Q + ques

    Q = Q + "The nodes in cluster "+i+" are: " + node + ". "
    Q = Q + "The egdes in cluster "+i+" are: " + edge + ". "
    Q = Q + "The exit nodes in cluster "+i+" are: " + exit + ". "

    Q = Q +"Q: You should give the shortest distances from node " + str(q[0])+" to all exit nodes and answer it started with 'Final Answer:'."
    #Q = Q + "Q: If these two target nodes, node " + str(q[0])+" and node " + str(q[1])+", are in the same cluster, please answer directly which cluster they are in."
    return Q

@retry(wait=wait_random_exponential(min=1, max=600), stop=stop_after_attempt(10))
def predict(Q, args):
    input = Q
    temperature = 0
    if args.SC == 1:
        temperature = 0.7

    Answer_list = []
    for text in input:
        # print(f"Sending request to DeepSeek API with text: {text}")
        # print("\n")
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text},
            ],
            temperature=temperature,
            max_tokens=args.token,
            stream=False,
            timeout=100 
        )
        Answer_list.append(response.choices[0].message.content.strip())
        print(f"response:{response.choices[0].message.content}")

    return Answer_list

def extract_cluster_info(text,j,args):  
    
    newpath = 'multi_gpt/shortest/intra_'+args.save_path+'_'+args.number+'/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    final_answer_start = text.find("final answer:")
    if final_answer_start != -1:
        text = text[final_answer_start:]
    else:
        final_answer_start = text.find("final answer")
        if final_answer_start !=-1:
            text = text[final_answer_start:]
        

    with open(newpath + "graph"+str(j)+"_"+args.number+".txt", "w", encoding="utf-8") as f:
            f.write(text)
        

def main():
    if 'OPENAI_API_KEY' in os.environ:
        openai.api_key = os.environ['OPENAI_API_KEY']
    else:
        raise Exception("Missing openai key!")
    if 'OPENAI_ORGANIZATION' in os.environ:
        openai.organization = os.environ['OPENAI_ORGANIZATION']
    res1,  res2, answer, answer_all = [], [], [], []

    g_num = args.end-args.begin #200

    batch_num = 1
    start_idx = args.begin-1

    file_path = 'data/shortest.json'
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f) 
    for i in tqdm(range((g_num + batch_num - 1) // batch_num)):

        G_list, Q_list, q_list,= [], [], []
        start_idx +=1
        for j in range(start_idx, start_idx+1):
            entry = data[j]
            Clusters_k='Cluster_'+args.number
            nodes_k='Exit_nodes'+args.number
            edges_k='Intra_'+args.number+'_edges'       
            if 'query_clustersplit' in entry:
                G= entry['query_clustersplit']
            if 'aim' in entry:
                q = entry['aim']
            if 'nodes' in entry:
                n = entry['nodes']
            if 'shortest' in entry:
                a = entry['shortest']
            if 'ques' in entry:
                ques = entry['ques']
            if Clusters_k in entry:
                node = entry[Clusters_k]
            if nodes_k in entry:
                exit = entry[nodes_k]
            if edges_k in entry:
                edge = entry[edges_k]

            Q = translate(G, q, args, j,ques,node,exit,edge,args.number)
            Q_list.append(Q)
            G_list.append(G)
            q_list.append(q)
            if j==0:
                print("/n")
            print(f"Processing batch {i + 1}, graph {j}")
        sc = 1
        if args.SC == 1:
            sc = args.SC_num
        sc_list = []
        for k in range(sc):
            answer_list = predict(Q_list, args)
            answer_to_text = ' '.join(answer_list)
            extract_cluster_info(answer_to_text.lower(),j,args)
            answer_all.append(answer_list) 
            sc_list.append(answer_list)

    res1, res2, answer = [0], [0], ["0"]
    merged = list(zip(res1, res2))
    res1 = np.array(res1)
    res2 = np.array(res2)
    answer = np.array(answer)
    answer_all = np.array(answer_all)
    print("res2")
    print(res2.sum()) 
    print(res1.sum()) 
    print("Answer: ", merged)

if __name__ == "__main__":
    main()
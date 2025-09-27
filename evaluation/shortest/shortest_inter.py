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
parser.add_argument('--begin', type=int, default=0, help='beginning')
parser.add_argument('--end', type=int, default=100, help='ending')
parser.add_argument('--source_path', type=str, default="None", help='path is must')
parser.add_argument('--split_path', type=str, default="None", help='path is must')
args = parser.parse_args()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def translate(G, q, args, number,edge):
    Q = ''
    prompt_folder = "prompt"

    if args.prompt in ["CoT", "k-shot","Algorithm","Instruct",'dot1','dot2','ins1','ins2','ins3','short']:
        with open("NLGraph/shortest_path/"+prompt_folder+"/" + args.prompt + "-prompt.txt", "r") as f:
            exemplar = f.read()
        Q = Q + exemplar + "\n\n\n"


    #Q = Q + G
    Q = Q + "The current graph is divided into two clusters.\n"

    Q = Q +"The shortest distances from node "+str(q[0])+" to the exit nodes in its cluster0 are as follows:"+'\n'

    file_path = "multi_gpt/shortest/intra_"+args.source_path+"_0/"+f"/graph{number}_0.txt"
    
    with open(file_path, "r", encoding="utf-8") as f:
        exemplar = f.read()
        Q = Q + exemplar + "\n"

    Q = Q +"The shortest distances from node "+str(q[1])+" to the exit nodes in its cluster1 are as follows:"+'\n'

    file_path = "multi_gpt/shortest/intra_"+args.source_path+"_1/"+f"/graph{number}_1.txt"
    
    with open(file_path, "r", encoding="utf-8") as f:
        exemplar = f.read()
        Q = Q + exemplar + "\n"
    
    Q = Q + "The edges that directly connect the two clusters are:"
    Q = Q + edge +'\n'

    Q = Q + "Q: Give the shortest path from node " + str(q[0])+" to node " + str(q[1])+" and answer it started with 'Final Answer:'."
    #Q = Q + "To find the shortest path from node "+ str(q[0])+" to node "+ str(q[1])+ ", you can evaluate all possible routes through their cluster exits and inter-cluster edges to determine the path with the minimal total weight."

    #ipdb.set_trace()


    return Q

def translate60(G, q, args, number,edge):
    Q = ''
    prompt_folder = "prompt"

    if args.prompt in ["CoT", "k-shot","Algorithm","Instruct",'dot1','dot2','ins1','ins2','ins3','short']:
        with open("NLGraph/shortest_path/"+prompt_folder+"/" + args.prompt + "-prompt.txt", "r") as f:
            exemplar = f.read()
        Q = Q + exemplar + "\n\n\n"


    #Q = Q + G
    Q = Q + "The current graph is divided into three clusters.\n"

    Q = Q +"The shortest distances from node "+str(q[0])+" to the exit nodes in its cluster0 are as follows:"+'\n'

    file_path = "multi_gpt/shortest/intra_"+args.source_path+"_0/"+f"/graph{number}_0.txt"
    
    with open(file_path, "r", encoding="utf-8") as f:
        exemplar = f.read()
        Q = Q + exemplar + "\n"

    Q = Q +"The shortest distances from node "+str(q[1])+" to the exit nodes in its cluster1 are as follows:"+'\n'

    file_path = "multi_gpt/shortest/intra_"+args.source_path+"_1/"+f"/graph{number}_1.txt"
    
    with open(file_path, "r", encoding="utf-8") as f:
        exemplar = f.read()
        Q = Q + exemplar + "\n"

    Q = Q +"The shortest distances between each exit nodes in cluster2 are as follows:"+'\n'

    file_path = "multi_gpt/shortest/intra_"+args.source_path+"_2/"+f"/graph{number}_2.txt"
    
    with open(file_path, "r", encoding="utf-8") as f:
        exemplar = f.read()
        Q = Q + exemplar + "\n"

    Q = Q + "The edges that directly connect the three clusters are:"
    Q = Q + edge +'\n'

    Q = Q + "Q: Give the shortest path from node " + str(q[0])+" to node " + str(q[1])+" and answer it started with 'Final Answer:'."
    
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
        Answer_list.append(response.choices[0].message.content)
        content = response.choices[0].message.content
        print(f"response: {content[-400:] if len(content) > 400 else content}")

    return Answer_list

def evaluate(ans, G, q, a):
    print("shortest GT is: ", a)
    entity = "node"
    mode_str = "the shortest path from " + entity + ' ' + str(q[0]) + " to " + entity + ' ' + str(q[1])
    pos = ans.find(mode_str)
    if pos == -1:
        mode_str = "the shortest paths from " + entity + ' ' + str(q[0]) + " to " + entity + ' ' + str(q[1])
        pos = ans.find(mode_str)
        if pos == -1:
            return 0, 0

    pos = ans.rfind("a total weight of")
    flag =1
    if pos == -1:
        pos = ans.rfind("the total shortest distance is")
        flag = 2
    if pos == -1:
        pos = ans.rfind("with a total path weight of")
        flag = 3
    if pos == -1:
        pos = ans.rfind("total weight")
        flag = 4
    if pos == -1:
        pos = ans.rfind("a total distance of")
        flag = 5
    if pos == -1:
        pos = ans.rfind("a total shortest distance of")
        flag = 6
    if pos == -1:
        return 0, 0

    scan_end = min(pos + 60, len(ans))
    eq_pos = ans.find("=", pos, scan_end)
    if eq_pos != -1:
        i = eq_pos + 2
        while i < len(ans) and not (ans[i] >= '0' and ans[i] <= '9'):
            i += 1
        num = 0
        while i < len(ans) and ans[i] >= '0' and ans[i] <= '9':
            num = num * 10 + int(ans[i])
            i += 1
    else:
        if flag ==1:
            i = pos + len("a total weight of")
        elif flag ==2:
            i = pos + len("the total shortest distance is")
        elif flag ==3:
            i = pos + len("with a total path weight of") 
        elif flag ==4:
            i = pos + len("total weight")
        elif flag ==5:
            i = pos + len("a total distance of")
        elif flag ==6:
            i = pos + len("a total shortest distance of")
        while i < len(ans) and not (ans[i] >= '0' and ans[i] <= '9'):
            i += 1
        num = 0
        while i < len(ans) and ans[i] >= '0' and ans[i] <= '9':
            num = num * 10 + int(ans[i])
            i += 1
    print("length: ", num)
    print("shortest: ", a)
    if num != a:
        return 1, 0
    return 1, 1

def main():
    if 'OPENAI_API_KEY' in os.environ:
        openai.api_key = os.environ['OPENAI_API_KEY']
    else:
        raise Exception("Missing openai key!")
    if 'OPENAI_ORGANIZATION' in os.environ:
        openai.organization = os.environ['OPENAI_ORGANIZATION']
    res1,  res2, answer, answer_all = [], [], [], []

    g_num = args.end-args.begin 
    

    batch_num = 1

    start_idx = args.begin-1

    file_path = 'data/shortest.json'
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)  
    for i in tqdm(range((g_num + batch_num - 1) // batch_num)):
        G_list, Q_list, q_list, a_list= [], [], [],[]
        start_idx +=1

        for j in range(start_idx, start_idx+1):
            entry = data[j]      
            if 'query_clustersplit' in entry:
                G= entry['query_clustersplit'] 
            if 'aim' in entry:
                q = entry['aim']
            if 'shortest' in entry:
                a = entry['shortest']
            if 'Cross_edges' in entry:
                edge = entry['Cross_edges']
            if 'nodes' in entry:
                n = entry['nodes']
            if n > 60:
                Q = translate60(G, q, args, j, edge)
            else:
                Q = translate(G, q, args, j, edge)

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
                    r1, r2 = evaluate(ans.lower(), G, q_list[j],a_list[j]) 
                    vote1 += r1
                    vote2 += r2
                except:
                    print(ans.lower())
            r1 = 1 if vote1*2 > sc else 0
            r2 = 1 if vote2*2 > sc else 0 
            res1.append(r1)
            res2.append(r2)

    merged = list(zip(res1, res2))
    res1 = np.array(res1)
    res2 = np.array(res2)
    answer = np.array(answer)

    print("res2 for answer right",res2.sum()) 
    print("res1 for extraction successful",res1.sum()) 
    print("Answer: ", merged)

if __name__ == "__main__":
    main()
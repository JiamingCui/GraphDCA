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
parser.add_argument('--token', type=int, default=4000, help='max token (default: 1000)')
parser.add_argument('--begin', type=int, default=0, help='beginning')
parser.add_argument('--end', type=int, default=100, help='ending')
args = parser.parse_args()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def translate(G, args):
    Q = ''
    with open("newdata/Graph-Reasoning-LLM-main/dataset/triangle.txt", "r") as f:
        exemplar = f.read()
    Q = Q + exemplar + "\n\n\n"

    Q = Q + G

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
        max_completion_tokens=args.token,
        stream=False,
        timeout=100
    )
    answer = response.choices[0].message.content
    print(f"response:{answer[:400] if len(answer) > 400 else answer}")
    return answer

def evaluate(ans, a):
    print("max GT is: ", a)
    start = ans.lower().find("final answer")
    if start == -1:
        return 0, 0
    text = ans[start:]

    keys = [
        "a total weight of",
        "the total shortest distance is",
        "with a total path weight of",
        "total weight",
        "a total distance of",
        "a total shortest distance of",
        "the maximum sum of the weights"
    ]
    
    num = None
    for k in keys:
        pos = text.find(k)
        if pos == -1:
            continue
        m = re.search(r'\d+', text[pos + len(k):])
        if m:
            num = int(m.group())
            break

    if num is None:
        m = re.search(r'=\s*(?:<<[^>]*>)?(\d+)', text)
        if m:
            num = int(m.group(1))

    if num is None:
        return 0, 0

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

    start_idx = args.begin - 1
    file_path = ''
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for i in tqdm(range(g_num)):
        idx = start_idx + i + 1
        if idx >= len(data):
            break

        entry = data[idx]
        
        if 'query' in entry:
            G = entry['query']
        if 'result' in entry:
            a = entry['result']
        
        Q = translate(G, args)
        
        print(f"\nProcessing graph {idx}...")

        answer_text = predict(Q, args)
        
        r1, r2 = evaluate(answer_text.lower(), a)
        res1.append(r1)
        res2.append(r2)
        
        
    
    
    
    if res1:
        res1 = np.array(res1)
        res2 = np.array(res2)
        print(f"\nTotal graphs processed: {len(res1)}")
        print("res2 for answer right", res2.sum())
        print("res1 for extraction successful", res1.sum())

if __name__ == "__main__":
    main()
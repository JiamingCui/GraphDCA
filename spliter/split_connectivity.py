import json
import re
import os
import random
import networkx as nx
from collections import defaultdict, deque


CNT = 0


def read_graph_from_json(data):
    edges = []
    query = data["query"]
    edge_pattern = re.compile(r'\((\d+), (\d+)\)')
    for u, v in edge_pattern.findall(query):
        edges.append((int(u), int(v)))
    return edges


def read_graph_from_json_clustersplit(data):
    edges = []
    query = data["query_clustersplit"]
    edge_pattern = re.compile(r'\((\d+),\s*(\d+)\)')
    for u, v in edge_pattern.findall(query):
        edges.append((int(u), int(v)))
    return edges


def find_shortest_path_length(edges, start, end):
    if start == end:
        return 0
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        node, dist = queue.popleft()
        for neighbor in graph[node]:
            if neighbor == end:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return float('inf')


def find_new_aim(edges, nodes, min_path_length, max_attempts=100):
    for _ in range(max_attempts):
        u, v = random.sample(nodes, 2)
        if find_shortest_path_length(edges, u, v) >= min_path_length:
            return [u, v]
    return None


def process_entry(data):
    global CNT
    if not all(k in data for k in ["query", "result", "aim"]):
        return data
    edges = read_graph_from_json(data)
    nodes = list({u for edge in edges for u in edge})
    if len(nodes) <= 30 or data["result"] != "Yes":
        return data
    u, v = data["aim"]
    if find_shortest_path_length(edges, u, v) > 5:
        return data
    for min_len in [6, 5, 4, 3, 2]:
        new_aim = find_new_aim(edges, nodes, min_len)
        if new_aim:
            data["aim"] = new_aim
            old_pattern = r'Is there a path between node \d+ and node \d+'
            data["query"] = re.sub(old_pattern,
                                   f'Is there a path between node {new_aim[0]} and node {new_aim[1]}',
                                   data["query"])
            CNT += 1
            break
    return data

def bfs_k_hop_partition(G, edges, a_node, b_node):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    A, B = {a_node}, {b_node}
    queue_a, queue_b = deque([a_node]), deque([b_node])
    node_partition = {a_node: 'A', b_node: 'B'}
    while queue_a or queue_b:
        if queue_a:
            cur = queue_a.popleft()
            for nb in adj[cur]:
                if nb not in node_partition:
                    node_partition[nb] = 'A'
                    A.add(nb)
                    queue_a.append(nb)
        if queue_b:
            cur = queue_b.popleft()
            for nb in adj[cur]:
                if nb not in node_partition:
                    node_partition[nb] = 'B'
                    B.add(nb)
                    queue_b.append(nb)
    unassigned = set(G.nodes()) - A - B
    for node in unassigned:
        (A if random.random() < 0.5 else B).add(node)
    return sorted(A), sorted(B)

def custom_json_dump(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('[\n')
        for i, entry in enumerate(data):
            f.write('  {\n')
            items = []
            for k, v in entry.items():
                items.append(f'    "{k}": {json.dumps(v, ensure_ascii=False)}')
            f.write(',\n'.join(items))
            f.write('\n  }' + (',' if i < len(data) - 1 else '') + '\n')
        f.write(']\n')


def main():
    file_path = 'datasets/train_set/connectivity_train.json'
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        if 'query' in entry:
            query = entry['query']
            nodes_match = re.search(r'nodes are numbered from \d+ to (\d+)', query)
            entry['nodes'] = int(nodes_match.group(1)) if nodes_match else None
            if 'answer' in entry:
                yes_no_match = re.search(r'### (Yes|No)\b', entry['answer'])
                entry['result'] = yes_no_match.group(1) if yes_no_match else None
            aim_match = re.search(r'Is there a path between node (\d+) and node (\d+)', query)
            entry['aim'] = [int(aim_match.group(1)), int(aim_match.group(2))] if aim_match else None
            split_match = re.split(r'Is there a path between', query)
            entry['query_clustersplit'] = split_match[0].strip() if len(split_match) > 1 else None

    connectivity1 = 'datasets/connectivity1.json'
    os.makedirs(os.path.dirname(connectivity1), exist_ok=True)
    with open(connectivity1, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Step1 done -> {connectivity1}")


    with open(connectivity1, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    processed_data2 = [process_entry(entry) for entry in data2]
    connectivity2 = 'datasets/connectivity2.json'
    with open(connectivity2, 'w', encoding='utf-8') as f:
        json.dump(processed_data2, f, indent=2, ensure_ascii=False)
    print(f"Step2 done -> {connectivity2}  (CNT={CNT})")


    with open(connectivity2, 'r', encoding='utf-8') as f:
        data3 = json.load(f)
    processed_data3 = []
    error_count = 0
    for i, entry in enumerate(data3):
        try:
            if 'nodes' not in entry or 'aim' not in entry:
                raise ValueError("Missing required fields")
            num_nodes = entry['nodes'] + 1
            edges = read_graph_from_json_clustersplit(entry)
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            G.add_edges_from(edges)
            a_node, b_node = entry['aim']
            if not (0 <= a_node < num_nodes and 0 <= b_node < num_nodes):
                raise ValueError(f"Aim nodes out of range: {entry['aim']}")
            cluster_0, cluster_1 = bfs_k_hop_partition(G, edges, a_node, b_node)
            exit_nodes0 = sorted(n for n in cluster_0 if any(nb in cluster_1 for nb in G.neighbors(n)))
            exit_nodes1 = sorted(n for n in cluster_1 if any(nb in cluster_0 for nb in G.neighbors(n)))
            intra0 = [tuple(sorted([u, v])) for u, v in edges if u in cluster_0 and v in cluster_0]
            intra1 = [tuple(sorted([u, v])) for u, v in edges if u in cluster_1 and v in cluster_1]
            cross = [tuple(sorted([u, v])) for u, v in edges
                     if (u in cluster_0 and v in cluster_1) or (u in cluster_1 and v in cluster_0)]
            new_entry = dict(entry)
            new_entry.update({
                "ques": "In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge.",
                "Cluster_0": str(cluster_0),
                "Cluster_1": str(cluster_1),
                "Exit_nodes0": str(exit_nodes0),
                "Exit_nodes1": str(exit_nodes1),
                "Intra_0_edges": str(intra0),
                "Intra_1_edges": str(intra1),
                "Cross_edges": str(cross),
            })
            processed_data3.append(new_entry)
            print(f"Processed entry {i}: nodes={num_nodes}, cut={len(cross)}")
        except Exception as e:
            error_count += 1
            print(f"Error entry {i}: {e}")
            continue
    connectivity = 'data/connectivity.json'
    custom_json_dump(processed_data3, connectivity)
    print(f"✅ Step3 done -> {connectivity}  (errors={error_count})")

if __name__ == "__main__":
    main()
import json
import re
import random
from collections import defaultdict, deque

edge_pattern = re.compile(r'\((\d+), (\d+)\)')

def parse_edges_from_query(query):
    return [(int(a), int(b)) for a, b in edge_pattern.findall(query)]

def has_cycle(n, edges):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    visited = [False] * n
    for start in range(n):
        if visited[start]:
            continue
        q = [(start, -1)]
        for node, parent in q:
            if visited[node]:
                continue
            visited[node] = True
            for nb in graph[node]:
                if nb == parent:
                    continue
                if visited[nb]:
                    return True
                q.append((nb, node))
    return False


def calc_k_score(clusters, edges):
    k = len(clusters)
    exits = [set() for _ in range(k)]
    intra_edges = [[] for _ in range(k)]
    cross_edges = []
    node2cluster = {v: idx for idx, s in enumerate(clusters) for v in s}
    for u, v in edges:
        cu, cv = node2cluster[u], node2cluster[v]
        if cu == cv:
            intra_edges[cu].append((u, v))
        else:
            cross_edges.append((u, v))
            exits[cu].add(u)
            exits[cv].add(v)
    return (
        sum(len(s) for s in exits),
        [sorted(s) for s in exits],
        intra_edges,
        cross_edges
    )

def bfs_grow_one(adj, start, quota, exclude):
    cluster = set()
    q = deque([start])
    cluster.add(start)
    exclude.add(start)
    while q and len(cluster) < quota:
        cur = q.popleft()
        for nb in adj[cur]:
            if nb not in exclude:
                cluster.add(nb)
                exclude.add(nb)
                q.append(nb)
                if len(cluster) >= quota:
                    break
        if len(cluster) >= quota:
            break
    return cluster


def best_k_partition(nodes, edges, k):
    nodes = list(nodes)
    n = len(nodes)
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for v in nodes:
        if v not in adj:
            adj[v] = []

    quota = n // k
    rem = n % k
    exclude = set()
    clusters = [set() for _ in range(k)]

    remaining_nodes = set(nodes)
    for i in range(k):
        if not remaining_nodes:
            break
        start = random.choice(list(remaining_nodes))
        size = quota + (1 if i < rem else 0)
        
        cluster_nodes = bfs_grow_one(adj, start, size, exclude)
        
        while len(cluster_nodes) < size and remaining_nodes - exclude:
            extra = random.choice(list(remaining_nodes - exclude))
            cluster_nodes.add(extra)
            exclude.add(extra)
        clusters[i] = cluster_nodes
        remaining_nodes = set(nodes) - exclude

    
    if remaining_nodes:
        clusters[-1].update(remaining_nodes)

    return [sorted(c) for c in clusters]


def normalize_entry(entry):
    edges = parse_edges_from_query(entry.get("query", ""))
    nodes_set = set(u for u, v in edges) | set(v for u, v in edges)
    entry["nodes"] = len(nodes_set)
    entry["result"] = "Yes" if has_cycle(max(nodes_set, default=-1) + 1, edges) else "No"

    if not nodes_set:
        entry.update({"Cross_edges": "[]"})
        return

    num_nodes = len(nodes_set)
    if 20 < num_nodes <= 40:
        k = 2
    elif 40 < num_nodes <= 60:
        k = 3
    elif 60 < num_nodes <= 80:
        k = 4
    elif num_nodes > 80:
        k = 5
    else:
        k = 2

    clusters = best_k_partition(nodes_set, edges, k)
    
    _, exits, intra_edges, cross_edges = calc_k_score(clusters, edges)

    
    for i in range(k):
        entry[f"Cluster_{i}"] = str(sorted(clusters[i]))
        entry[f"Exit_nodes{i}"] = str(exits[i])
        entry[f"Intra_{i}_edges"] = str(intra_edges[i])
    entry["Cross_edges"] = str(cross_edges)


def process_json(json_path, new_path):
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    for i, entry in enumerate(data[0:3000], 1):
        print(f"Processing entry {i} in {json_path}")
        normalize_entry(entry)
    with open(new_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("✅ save as", new_path)

if __name__ == "__main__":
    input_json = 'datasets/train_set/cycle_train.json'
    output_json = 'data/cycle.json'
    process_json(input_json, output_json)
import json
import re
import os
import heapq
import random
import networkx as nx
from collections import defaultdict, deque


edge_pattern = re.compile(r'\((\d+),(\d+),(\d+)\)')
def read_graph_from_json(data):
    query = data["query_clustersplit"]
    return [(int(u), int(v), int(w)) for u, v, w in edge_pattern.findall(query)]


def dijkstra_shortest_path(G, start, end):
    heap = [(0, start)]
    dist = {start: 0}
    prev = {start: None}
    while heap:
        d, u = heapq.heappop(heap)
        if u == end:
            break
        for v in G.neighbors(u):
            nd = d + G[u][v]["weight"]
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if end not in dist:
        return None, []
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return dist[end], path


def multi_2_bfs_partition(G, sources, all_nodes):
    k = len(sources)
    clusters = [set() for _ in range(k)]
    q = deque((s, i) for i, s in enumerate(sources))
    for s, i in q:
        clusters[i].add(s)
    adj = {n: list(G.neighbors(n)) for n in G.nodes}
    while q:
        cur, idx = q.popleft()
        for nb in adj[cur]:
            if not any(nb in c for c in clusters):
                clusters[idx].add(nb)
                q.append((nb, idx))
    assigned = set().union(*clusters)
    for n in all_nodes - assigned:
        clusters[random.randrange(k)].add(n)
    return [sorted(c) for c in clusters]

def multi_3_bfs_partition(G, sources, all_nodes):
    assert len(sources) == 2
    one_third = len(all_nodes) / 3.0
    clusters = [set(), set(), set()]
    q0, q1 = deque([sources[0]]), deque([sources[1]])
    clusters[0].add(sources[0])
    clusters[1].add(sources[1])
    adj = {n: list(G.neighbors(n)) for n in G.nodes}
    while q0 or q1:
        if q0 and len(clusters[0]) < one_third:
            cur = q0.popleft()
            for nb in adj[cur]:
                if nb not in clusters[0] and nb not in clusters[1]:
                    clusters[0].add(nb)
                    q0.append(nb)
        if q1 and len(clusters[1]) < one_third:
            cur = q1.popleft()
            for nb in adj[cur]:
                if nb not in clusters[0] and nb not in clusters[1]:
                    clusters[1].add(nb)
                    q1.append(nb)
        if len(clusters[0]) >= one_third and len(clusters[1]) >= one_third:
            break
    assigned = clusters[0] | clusters[1]
    clusters[2] = all_nodes - assigned
    return [sorted(c) for c in clusters]


def classify_edges(G, clusters):
    k = len(clusters)
    node2c = {n: i for i, c in enumerate(clusters) for n in c}
    intra = [[] for _ in range(k)]
    cross = []
    for u, v, w in G.edges(data='weight'):
        cu, cv = node2c[u], node2c[v]
        if cu == cv:
            intra[cu].append((u, v, w))
        else:
            cross.append((u, v, w))
    exits = [
        sorted({n for n in c if any(nb in G and nb in node2c and node2c[nb] != i for nb in G.neighbors(n))})
        for i, c in enumerate(clusters)
    ]
    return intra, cross, exits


def process_entry(entry):
    try:
        edges = read_graph_from_json(entry)
        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        G.add_nodes_from(range(entry['nodes']))
        start, end = entry['aim']
        min_dist, path = dijkstra_shortest_path(G, start, end)
        entry['shortest'] = min_dist
        entry['result'] = path
        entry['ques'] = "In an undirected graph, (i,j,k) means that node i and node j are connected with an undirected edge with weight k. "
        k = 3 if entry['nodes'] > 60 else 2
        sources = [start, end]
        if k == 2:
            clusters = multi_2_bfs_partition(G, sources, set(G.nodes))
        else:
            clusters = multi_3_bfs_partition(G, sources, set(G.nodes))
        intra, cross, exits = classify_edges(G, clusters)
        for i in range(k):
            entry[f'Cluster_{i}'] = str(clusters[i])
            entry[f'Exit_nodes{i}'] = str(exits[i])
            entry[f'Intra_{i}_edges'] = str([tuple(e) for e in intra[i]])
        entry['Cross_edges'] = str([tuple(e) for e in cross])
    except Exception as e:
        print(f"Error entry {entry.get('index')}: {e}")
        raise


def custom_json_dump(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('[\n')
        for idx, entry in enumerate(data):
            f.write('  {\n')
            f.write(',\n'.join(f'    "{k}": {json.dumps(v, ensure_ascii=False)}'
                               for k, v in entry.items()))
            f.write('\n  }' + (',' if idx < len(data) - 1 else '') + '\n')
        f.write(']\n')


def main():

    file_path = 'datasets/train_set/shortest_train.json'
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        if 'query' in entry:
            query = entry['query']
            nodes_match = re.search(r'nodes are numbered from \d+ to (\d+)', query)
            entry['nodes'] = int(nodes_match.group(1)) if nodes_match else None
            if 'answer' in entry:
                shortest_match = re.search(r'### (\d+)', entry['answer'])
                entry['shortest'] = int(shortest_match.group(1)) if shortest_match else None
            aim_match = re.search(r'Give the weight of the shortest path from node (\d+) to node (\d+)', query)
            entry['aim'] = [int(aim_match.group(1)), int(aim_match.group(2))] if aim_match else None
            split_match = re.split(r'Give the weight of the shortest path', query)
            entry['query_clustersplit'] = split_match[0].strip() if len(split_match) > 1 else None
    os.makedirs('newdata/data', exist_ok=True)
    step1 = 'datasets/shortest1.json'
    with open(step1, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Step1 done -> {step1}")


    with open(step1, encoding='utf-8') as f:
        data2 = json.load(f)
    cnt = 0
    for e in data2:
        process_entry(e)
        print(f"Processed entry {cnt}")
        cnt += 1
    step2 = 'data/shortest.json'
    custom_json_dump(data2, step2)
    print(f"✅ Step2 done -> {step2}")

if __name__ == "__main__":
    main()
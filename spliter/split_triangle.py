import json
import re
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np


EDGE_PATTERN = re.compile(r"\((\d+), (\d+)\)")
NODE_PATTERN = re.compile(r"\[(\d+), (\d+)\]")


def parse_edges_from_query(query: str) -> List[Tuple[int, int]]:
	return [(int(u), int(v)) for u, v in EDGE_PATTERN.findall(query)]


def parse_nodes_from_query(query: str) -> List[Tuple[int, int]]:
	return [(int(u), int(w)) for u, w in NODE_PATTERN.findall(query)]


def normalize_entry(entry: Dict) -> None:
	edges = parse_edges_from_query(entry.get("query", ""))
	nodes_set = {u for u, _ in edges} | {v for _, v in edges}
	entry["nodes"] = len(nodes_set)

	if "answer" in entry:
		result_match = re.search(r"### (\d+)", entry["answer"])
		entry["result"] = int(result_match.group(1)) if result_match else None


def classify_edges(
	graph: nx.Graph, clusters: List[List[int]]
) -> Tuple[List[List[Tuple[int, int]]], List[Tuple[int, int]], List[List[int]]]:
	node_to_cluster = {n: i for i, cluster in enumerate(clusters) for n in cluster}
	intra_edges = [[] for _ in range(len(clusters))]
	cross_edges: List[Tuple[int, int]] = []

	for u, v in graph.edges:
		cu, cv = node_to_cluster[u], node_to_cluster[v]
		if cu == cv:
			intra_edges[cu].append((u, v))
		else:
			cross_edges.append((u, v))

	exits: List[List[int]] = []
	for i, cluster in enumerate(clusters):
		current = {
			n
			for n in cluster
			if any(node_to_cluster.get(nb) != i for nb in graph.neighbors(n))
		}
		exits.append(sorted(current))

	return intra_edges, cross_edges, exits


def spectral_split(graph: nx.Graph, all_nodes: List[int]) -> List[List[int]]:
	nodes = sorted(all_nodes)
	n = len(nodes)

	if n <= 1:
		return [nodes, []]

	node_to_idx = {node: i for i, node in enumerate(nodes)}
	adjacency = np.zeros((n, n))
	for u, v in graph.edges():
		i, j = node_to_idx[u], node_to_idx[v]
		adjacency[i, j] = 1
		adjacency[j, i] = 1

	degree = np.sum(adjacency, axis=1)
	two_m = np.sum(degree)
	if two_m == 0:
		mid = n // 2
		return [nodes[:mid], nodes[mid:]]

	modularity = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			modularity[i, j] = adjacency[i, j] - degree[i] * degree[j] / two_m

	eigenvalues, eigenvectors = np.linalg.eigh(modularity)
	idx_max = int(np.argmax(eigenvalues))
	max_eval = eigenvalues[idx_max]
	vec = eigenvectors[:, idx_max]

	if max_eval <= 1e-10:
		if len(eigenvalues) > 1:
			idx_second = int(np.argsort(eigenvalues)[-2])
			if eigenvalues[idx_second] > 1e-10:
				vec = eigenvectors[:, idx_second]
			else:
				mid = n // 2
				return [nodes[:mid], nodes[mid:]]
		else:
			mid = n // 2
			return [nodes[:mid], nodes[mid:]]

	group1 = [nodes[i] for i in np.where(vec > 0)[0]]
	group2 = [nodes[i] for i in np.where(vec <= 0)[0]]

	if not group1 and group2:
		group1.append(group2.pop())
	elif not group2 and group1:
		group2.append(group1.pop())

	return [sorted(group1), sorted(group2)]


def enrich_entry(entry: Dict) -> None:
	edges = parse_edges_from_query(entry["query"])
	graph = nx.Graph()
	graph.add_edges_from(edges)
	graph.add_nodes_from(range(entry["nodes"]))

	node_to_weight = dict(parse_nodes_from_query(entry["query"]))
	entry["ques"] = (
		"In an undirected graph, [i, k] means that node i has the weight k. "
		"(i,j) means that node i and node j are connected with an undirected edge. "
	)

	clusters = spectral_split(graph, list(graph.nodes))
	intra, cross, exits = classify_edges(graph, clusters)

	for i, cluster in enumerate(clusters):
		entry[f"Cluster_{i}"] = str([[n, node_to_weight[n]] for n in cluster])
		entry[f"Exit_nodes{i}"] = str(exits[i])
		entry[f"Intra_{i}_edges"] = str([tuple(e) for e in intra[i]])

	entry["Cross_edges"] = str([tuple(e) for e in cross])
	all_exit_nodes = sorted({n for group in exits for n in group})
	entry["All_exit_nodes"] = str([[n, node_to_weight[n]] for n in all_exit_nodes])


def custom_json_dump(data: List[Dict], output_path: str) -> None:
	with open(output_path, "w", encoding="utf-8") as f:
		f.write("[\n")
		for idx, entry in enumerate(data):
			f.write("  {\n")
			f.write(",\n".join(f'    "{k}": {json.dumps(v, ensure_ascii=False)}' for k, v in entry.items()))
			f.write("\n  }" + ("," if idx < len(data) - 1 else "") + "\n")
		f.write("]\n")


def process_json(input_path: str, output_path: str) -> None:
	with open(input_path, encoding="utf-8") as f:
		data = json.load(f)

	for i, entry in enumerate(data, 1):
		print(f"Processing entry {i}")
		normalize_entry(entry)
		enrich_entry(entry)

	custom_json_dump(data, output_path)
	print(f"Saved to {output_path}")


if __name__ == "__main__":
	input_json = "datasets/train_set/triangle_train.json"
	output_json = "data/triangle.json"
	process_json(input_json, output_json)

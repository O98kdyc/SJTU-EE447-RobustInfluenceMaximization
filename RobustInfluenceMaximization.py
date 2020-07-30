import numpy as np
from copy import deepcopy
import random

class Edge:
    def __init__(self, src, dst, pro):
        self.src = src
        self.dst = dst
        self.pro = pro

class Graph:
    def __init__(self, num_nodes, delta=1):
        self.num_nodes = num_nodes
        self.delta = delta
        self.node_arr = [nid for nid in range(num_nodes)]
        self.edge_dict_arr_sd = [dict() for _ in range(num_nodes)]
        self.edge_dict_arr_ds = [dict() for _ in range(num_nodes)]

    def add_edge(self, src, dst, weight):
        if src in self.edge_dict_arr_ds[dst]:
            print(f"Edge {src}->{dst} exists")
            raise AssertionError()
        else:
            edge = Edge(src, dst, weight)
            self.edge_dict_arr_sd[src][dst] = edge
            self.edge_dict_arr_ds[dst][src] = edge

    def read_from_file(self, fp, undirected=False):
        with open(fp, 'r') as f:
            node = []
            for line in f.readlines():
                if line[0] != '#':
                    dst, src, weight = line.split()
                    src, dst, weight = int(src), int(dst), float(weight)
                    if dst not in node:
                        node.append(dst)
                    if src not in node:
                        node.append(src)

                    self.add_edge(src, dst, weight)
                    if undirected:
                        self.add_edge(dst, src, weight)
            #print(len(node))

    def expect_ic_influence(self, seed_arr, max_iter = 100):
        total = 0
        for _ in range(max_iter):
            total = total + self.ic_influence(seed_arr)
        return total/max_iter

    def ic_influence(self, seed_arr):
        seed_arr = list(seed_arr)

        node_visited_arr = np.array([False] * self.num_nodes)
        node_visited_arr[seed_arr] = True

        bfs_nid_arr = deepcopy(seed_arr)
        current_round = 0
        last_node = []
        last_node.append(bfs_nid_arr[-1])
        while len(bfs_nid_arr) > 0:
            cur_nid = bfs_nid_arr.pop(0)
            flag = False
            for dst_nid, edge in self.edge_dict_arr_sd[cur_nid].items():
                tmp = 0
                flag = True
                if random.random() < edge.pro * (self.delta ** current_round):
                    if not node_visited_arr[dst_nid]:
                        node_visited_arr[dst_nid] = True
                        bfs_nid_arr.append(dst_nid)
                        tmp = dst_nid
            if cur_nid == last_node[current_round] and flag:
                current_round = current_round + 1
                last_node.append(tmp)

        return node_visited_arr.sum()

    def influence_estimation(self, seed_arr, max_iter = 10):
        seed_arr = list(seed_arr)
        pro_arr = np.zeros((max_iter, self.num_nodes))
        pro_arr[0][seed_arr] = 1
        for i in range(1, max_iter):
            for nid in range(self.num_nodes):
                not_before_pro = 1
                for j in range(i):
                    not_before_pro = not_before_pro - pro_arr[j][nid]
                not_i_pro = 1
                for src_nid, edge in self.edge_dict_arr_ds[nid].items():
                    not_i_pro = not_i_pro * (1 - pro_arr[i-1][src_nid] * edge.pro * (self.delta**(i-1)))
                pro_arr[i][nid] = not_before_pro * (1 - not_i_pro)

        ex_number = pro_arr.sum()
        return ex_number

    def max_margin_influence(self, existing_node, not_node):
        existing_node = list(existing_node)
        not_node = list(not_node)
        influence = 0
        max_nid = 0
        for nid in range(self.num_nodes):
            if (nid in existing_node) or (nid in not_node):
                continue
            seed = deepcopy(existing_node)
            seed.append(nid)
            #print("seed", seed)
            tmp = self.influence_estimation(seed)
            if tmp > influence:
                influence = tmp
                max_nid = nid
        return max_nid

    def max_influence_node(self, existing_node):
        influence = 0
        max_nid = 0
        for nid in range(self.num_nodes):
            if nid in existing_node:
                continue
            seed = []
            seed.append(nid)
            tmp = self.influence_estimation(seed)
            if tmp > influence:
                influence = tmp
                max_nid = nid
        return max_nid

    def robust_seed_selection(self, max_node_number, max_fail_number):
        if (max_node_number < max_fail_number + 1):
            raise AssertionError()
        exist, seed_arr = [], []
        for _ in range(max_fail_number):
            nid = self.max_influence_node(exist)
            exist.append(nid)

        for _ in range(max_node_number - max_fail_number):
            nid = self.max_margin_influence(seed_arr, exist)
            seed_arr.append(nid)

        return exist, seed_arr

g = Graph(348, 0.6)
g.read_from_file("fc.txt")
nodes = [i for i in range(348)]

tmp, robust_seed = g.robust_seed_selection(10, 0)
robust_seed.extend(tmp)
print(robust_seed)

k_m = 6
robust_seed = np.random.choice(robust_seed, k_m, replace = False)
influence = g.ic_influence(robust_seed)
print("Seed Test Influence: ", influence)
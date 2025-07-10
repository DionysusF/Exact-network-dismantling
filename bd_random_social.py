import networkx as nx
from copy import deepcopy
import time
import pandas as pd


def largestHD(G,s):
    global number_of_remove
    largest_component = max(nx.connected_components(G), key=len)
    if len(largest_component) > s:
        j = 0
        v = 0
        for i in largest_component:
            if G.degree(i) > j:
                j = G.degree(i)
                v = i
        number_of_remove += 1
        G_out = deepcopy(G)
        G_out.remove_node(v)
        return largestHD(G_out,s)
    if len(largest_component) <= s:
        return number_of_remove

def read_graph_from_file(file_path):
    # 初始化一个空字典来存储图信息
    # 字典的键是顶点，值是与该顶点相连的顶点集合
    graph = {}

    # 打开文件
    with open(file_path, 'r') as file:
        # 逐行读取文件
        for line in file:
            # 去除行尾的换行符，并以空格分割字符串，提取顶点信息
            vertex1, vertex2, _ = line.strip().split(' ')

            # 将字符串转换为整数
            vertex1, vertex2 = int(vertex1), int(vertex2)

            # 将边信息添加到图中
            if vertex1 not in graph:
                graph[vertex1] = set()
            if vertex2 not in graph:
                graph[vertex2] = set()

            # 由于是无向图，所以两个顶点都需要记录对方
            graph[vertex1].add(vertex2)
            graph[vertex2].add(vertex1)

    return graph
def create_networkx_graph(graph_dict):
    # 创建一个空的无向图
    G = nx.Graph()

    # 添加边
    for vertex, neighbors in graph_dict.items():
        for neighbor in neighbors:
            G.add_edge(vertex, neighbor)

    return G

def find_cc(G, node, size_Lcc):
    connected_component = []
    cluster = [node]
    def find_cluster_neighbors(G, cluster):
        cluster_neighbors = set()
        cluster_copy = set(cluster)
        for node in cluster_copy:
            neighbors = set(G.neighbors(node)) - cluster_copy
            cluster_neighbors |= neighbors
        return cluster_neighbors

    def find_connected_component(cluster, size_Lcc, selected_node=set()):
        nonlocal G
        nonlocal connected_component

        if len(cluster) == size_Lcc:
            connected_component.append(deepcopy(cluster))
        else:
            cluster_neighbors = find_cluster_neighbors(G, cluster)
            cluster_neighbors -= selected_node
            cluster_copy = cluster.copy()
            selected_node_copy = selected_node.copy()
            for node_ii in cluster_neighbors:
                cluster_copy.append(node_ii)
                selected_node_copy.add(node_ii)
                find_connected_component(cluster_copy, size_Lcc, selected_node_copy)
                cluster_copy.pop()
        return 0

    find_connected_component(cluster, size_Lcc)
    return connected_component


def branch_and_bound(Gr, s, d, dismantling_set, cut_number, cut_opt):
    global min_cut_number
    global min_dismantling_set
    global counting
    largest_component = max(nx.connected_components(Gr), key=len)
    if len(largest_component) > s:
        j = 0
        v = 0
        for i in largest_component:
            if Gr.degree(i) > j:
                j = Gr.degree(i)
                v = i
        all_allocation = []
        for j in range(s, 0, -1):
            all_allocation += find_cc(Gr, v, j)
        for id in range(0, len(all_allocation)+1):
            if id == 0:
                cut_number += 1
                if cut_number > min_cut_number:
                    counting += 1
                    break


                else:
                    G_out = deepcopy(Gr)
                    G_out.remove_node(v)
                    dismantling_set.append(v)
                    counting += 1
                    branch_and_bound(G_out, s, d+1, dismantling_set, cut_number, cut_opt)
                    dismantling_set.remove(v)
                cut_number -= 1

            else:
                sub_neighbors = []
                if id == len(all_allocation):
                    sub_neighbors = list(Gr.neighbors(v))
                else:
                    for sub_node in all_allocation[id - 1]:
                        for sub_node_nei in list(Gr.neighbors(sub_node)):
                            sub_neighbors.append(sub_node_nei)
                    sub_neighbors = set(sub_neighbors)
                    for x in all_allocation[id - 1]:
                        sub_neighbors.remove(x)

                cut_number += len(sub_neighbors)
                if cut_number > min_cut_number or len(sub_neighbors) >= cut_opt:

                    counting += 1
                else:
                    G_out = deepcopy(Gr)
                    for k in sub_neighbors:
                        dismantling_set.append(k)
                        G_out.remove_node(k)
                    counting += 1
                    branch_and_bound(G_out, s, d + 1, dismantling_set, cut_number,  cut_opt)
                    for k in sub_neighbors:
                        dismantling_set.remove(k)
                cut_number -= len(sub_neighbors)


    else:
        if cut_number == min_cut_number:
            counting += 1
            min_cut_number = cut_number
            min_dismantling_set.append(deepcopy(dismantling_set))
        else:
            print(counting, len(dismantling_set), dismantling_set)
            counting += 1
            min_cut_number = cut_number
            min_dismantling_set = [deepcopy(dismantling_set)]

def branch_and_bound_size(Gr, s, d, dismantling_set, cut_number, cut_opt):
    global min_cut_number
    global min_dismantling_set
    global counting
    largest_component = max(nx.connected_components(Gr), key=len)
    if len(largest_component) > s:
        j = 0
        v = 0
        for i in largest_component:
            if Gr.degree(i) > j:
                j = Gr.degree(i)
                v = i
        all_allocation = []
        for j in range(s, 0, -1):
            all_allocation += find_cc(Gr, v, j)
        for id in range(0, len(all_allocation)+1):
            if id == 0:
                cut_number += 1
                if cut_number >= min_cut_number:
                    counting += 1
                    break


                else:
                    G_out = deepcopy(Gr)
                    G_out.remove_node(v)
                    dismantling_set.append(v)
                    counting += 1
                    branch_and_bound_size(G_out, s, d+1, dismantling_set, cut_number, cut_opt)
                    dismantling_set.remove(v)
                cut_number -= 1

            else:
                sub_neighbors = []
                if id == len(all_allocation):
                    sub_neighbors = list(Gr.neighbors(v))
                else:
                    for sub_node in all_allocation[id - 1]:
                        for sub_node_nei in list(Gr.neighbors(sub_node)):
                            sub_neighbors.append(sub_node_nei)
                    sub_neighbors = set(sub_neighbors)
                    for x in all_allocation[id - 1]:
                        sub_neighbors.remove(x)

                cut_number += len(sub_neighbors)
                if cut_number >= min_cut_number or len(sub_neighbors) >= cut_opt:

                    counting += 1
                else:
                    G_out = deepcopy(Gr)
                    for k in sub_neighbors:
                        dismantling_set.append(k)
                        G_out.remove_node(k)
                    counting += 1
                    branch_and_bound_size(G_out, s, d + 1, dismantling_set, cut_number,  cut_opt)
                    for k in sub_neighbors:
                        dismantling_set.remove(k)
                cut_number -= len(sub_neighbors)


    else:
        if cut_number == min_cut_number:
            counting += 1
            min_cut_number = cut_number
            min_dismantling_set.append(deepcopy(dismantling_set))
        else:
            print(counting, len(dismantling_set), dismantling_set)
            counting += 1
            min_cut_number = cut_number
            min_dismantling_set = [deepcopy(dismantling_set)]


def main(args):
    global min_cut_number
    global min_dismantling_set
    global counting
    start = time.perf_counter()
    min_dismantling_set = []
    dismantling_set = []
    min_cut_number = args.n

    counting = 0
    if args.type == 'ER':
        p = args.ave_degree / (args.n - 1)
        G = nx.gnp_random_graph(args.n, p, seed=args.seed)
    if args.type == 'RR':
        G = nx.random_regular_graph(args.ave_degree, args.n, seed=args.seed)
    if args.type == 'BA':
        G = nx.barabasi_albert_graph(n=args.n, m=args.BA_m, seed=args.seed)
    if args.type == 'social_network':
        if args.name == "911":
            file_path = './social_networks/911.txt'
            graph = read_graph_from_file(file_path)
            G = create_networkx_graph(graph)
        elif args.name == "karate":
            G = nx.karate_club_graph()
        else:
            gml_file = f'./social_networks/{args.name}.gml'
            G = nx.read_gml(gml_file)
    print(G)
    G_copy = deepcopy(G)
    cut_number = 0
    for pre_delete in range(0, args.num_of_pre_delete):
        pre_j = 0
        pre_v = 0
        largest_component = max(nx.connected_components(G_copy), key=len)
        for i_pre in largest_component:
            if G_copy.degree(i_pre) > pre_j:
                pre_j = G_copy.degree(i_pre)
                pre_v = i_pre
        G_copy.remove_node(pre_v)
        dismantling_set.append(pre_v)
        cut_number += 1
    if args.mode == "size_only":
        print("size")
        branch_and_bound_size(G_copy, args.C, 0, [], cut_number, args.cut_opt)

    if args.mode == "all_MDS":
        print("all_MDS")
        branch_and_bound(G_copy, args.C, 0, [], cut_number, args.cut_opt)


    print('min_cut_number' + str(min_cut_number), 'dismantling_set' + str(min_dismantling_set))
    print('counting', counting)
    print('k' + str((2 ** len(G)) / counting))
    end = time.perf_counter()
    print("time", end - start)

    return end - start, (2 ** len(G)) / counting, min_cut_number, min_dismantling_set

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-mode", type=str, default="size_only", help="size_only,all_MDS")
    parser.add_argument("-type", type=str, default="ER", help="ER,RR,BA,social_network")
    parser.add_argument("-name", type=str, default="911", help="karate,polbooks,football,lesmis,celegansneural,911")
    parser.add_argument("-n", type=int, default=60, help="number of vertices")
    parser.add_argument("-ave_degree", type=int, default=3.5, help="number of vertices")
    parser.add_argument("-C", type=int, default=3, help="C-dismantling")
    parser.add_argument("-BA_m", type=int, default=3, help="m in BA graph")
    parser.add_argument("-seed", type=int, default=1, help="seed")
    parser.add_argument("-num_of_pre_delete", type=int, default=0, help="num_of_pre_delete")
    parser.add_argument("-cut_opt", type=int, default=100, help="cut the branch if neighbors of group greater than cut_opt")
    args = parser.parse_args()
    time, k, min_cut_number, min_dismantling_set = main(args)









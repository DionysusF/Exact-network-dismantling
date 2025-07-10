import networkx as nx
from copy import deepcopy
import time
import pandas as pd
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

Y = []
def corHD(G,s):

    number_of_remove = 0
    while len(max(nx.connected_components(G), key=len)) > s:
        if len(list(nx.k_core(G,2))) != 0:
            # print(number_of_remove,'no')
            j = 0
            v = None
            for ii in list(nx.k_core(G, 2)):
                # print(ii,G.degree(ii))
                if int(G.degree(ii)) > j:
                    j = G.degree(ii)
                    v = ii
            number_of_remove += 1
            # print('yes')
            G.remove_node(v)
        else:
            # print(number_of_remove,'yes')
            j = 0
            v = 0
            # for i in list(G):
            for i in list(max(nx.connected_components(G), key=len)):
                if G.degree(i) > j:
                    j = G.degree(i)
                    v = i
            number_of_remove += 1

            G.remove_node(v)

    return number_of_remove

def main(args):
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
    num_of_remove = corHD(G, args.C)
    print('len_of_dis_set:', num_of_remove)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-n", type=int, default=100, help="number of vertices")
    parser.add_argument("-ave_degree", type=int, default=3.5, help="number of vertices")
    parser.add_argument("-C", type=int, default=5, help="C-dismantling")
    parser.add_argument("-BA_m", type=int, default=4, help="m in BA graph")
    parser.add_argument("-type", type=str, default="ER", help="ER,RR,BA,social_network")
    parser.add_argument("-name", type=str, default="911", help="karate,polbooks,football,lesmis,celegansneural,911")
    parser.add_argument("-seed", type=int, default=5, help="seed")
    args = parser.parse_args()
    num_of_remove = main(args)

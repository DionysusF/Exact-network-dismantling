import networkx as nx
from copy import deepcopy
import time
import pandas as pd
# csv_file = "E:/PycharmProjects/pythonProject/venv/share/edges.csv"
# edge_data = pd.read_csv(csv_file)

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

def nodes_on_surface(G, center_node, radius):
    if radius == 0:
        result = [center_node]
    else:
        visited = set()
        queue = [(center_node, 0)]
        result = []

        while queue:
            node, depth = queue.pop(0)
            visited.add(node)

            if depth == radius:
                result.append(node)
            elif depth < radius:
                for neighbor in G.neighbors(node):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
        result = list(set(result))



    return result
def CI(G,node,l):
    if l == 0:
        result = G.degree(node) - 1
    else:
        result = G.degree(node) - 1
        ball = nodes_on_surface(G, node, l)
        for ii in ball:
            result *= G.degree(ii) - 1

    return result




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
    best_num_set = 100000000
    best_l = 0
    best_set = []
    for l in range(args.l_start, args.l_end + 1):
        G_copy = deepcopy(G)
        dis_set = []
        while len(max(nx.connected_components(G_copy), key=len)) > args.C:
            max_ci = -1
            max_ci_node = 0
            # print(list(G))
            for ii in list(G_copy):
                ci = CI(G_copy, ii, l)
                if ci > max_ci:
                    max_ci = ci
                    max_ci_node = ii
            # print(max_ci_node)
            dis_set.append(max_ci_node)
            G_copy.remove_node(max_ci_node)
        if len(dis_set) < best_num_set:
            best_num_set = len(dis_set)
            best_set = dis_set
            best_l = l

    print(len(best_set), 'l:', best_l, best_set, )
    return best_l, len(best_set)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-n", type=int, default=100, help="number of vertices")
    parser.add_argument("-ave_degree", type=int, default=3.5, help="number of vertices")
    parser.add_argument("-C", type=int, default=5, help="C-dismantling")
    parser.add_argument("-BA_m", type=int, default=1, help="m in BA graph")
    parser.add_argument("-type", type=str, default="ER", help="ER,RR,BA,social_network")
    parser.add_argument("-name", type=str, default="911", help="karate,polbooks,football,lesmis,celegansneural,911")
    parser.add_argument("-seed", type=int, default=5, help="seed")
    parser.add_argument("-l_start", type=int,default=1, help="l_start")
    parser.add_argument("-l_end", type=int,default=30, help="l_end")
    args = parser.parse_args()
    best_l, len_best_set = main(args)




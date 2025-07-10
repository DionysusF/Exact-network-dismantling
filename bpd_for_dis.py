
import networkx as nx
import math
from copy import deepcopy
import numpy as np
import random
import pandas as pd
import time
from random import seed
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

def zij(i,j,x,omega_i,g,arry):
    cav = list(nx.neighbors(g,i))
    cav.remove(j)
    fir = 1
    for nei_c in cav:
        fir *= arry[nei_c][i][0]+arry[nei_c][i][1]
    sec = 1
    for nei_c in cav:
        cav_1 = deepcopy(cav)
        cav_1.remove(nei_c)
        sec_sec = 1
        for node_1 in cav_1:
            sec_sec *= arry[node_1][i][0]+arry[node_1][i][1]
        sec += (1-arry[nei_c][i][0])*sec_sec
    z_i_j = 1+(math.exp(x*omega_i))*(fir+sec)
    q0_i_j = 1/z_i_j
    qi_i_j = (math.exp(x*omega_i))*fir/z_i_j
    return [z_i_j,q0_i_j,qi_i_j]

def zi(i,x,omega_i,g,arry):
    cav = list(nx.neighbors(g, i))
    fir = 1
    for nei_c in cav:
        fir *= arry[nei_c][i][0] + arry[nei_c][i][1]
    sec = 1
    for nei_c in cav:
        cav_1 = deepcopy(cav)
        cav_1.remove(nei_c)
        sec_sec = 1
        for node_1 in cav_1:
            sec_sec *= arry[node_1][i][0] + arry[node_1][i][1]
        sec += (1 - arry[nei_c][i][0]) * sec_sec
    z_i= 1 + math.exp(x * omega_i) * (fir + sec)
    q0_i = 1 / z_i

    return q0_i

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

    omega_i = args.omega_i

    T = args.T
    arry_ori = []
    s = args.C

    lis = []
    for i in range(0, len(G)):
        lis += [i]
    random.shuffle(lis)
    min_set = []
    min_set_num = 100000000
    for x in np.arange(args.x_start, args.x_end + args.x_step, args.x_step):
        fvs = []
        G_copy = deepcopy(G)
        for i in range(0, len(G)):
            arry_ori += [[]]
            for j in range(0, len(G)):
                arry_ori[i] += [[0, 0]]
        for i in range(0, len(G)):
            for j in nx.neighbors(G_copy, i):
                a = random.uniform(0, 1)
                b = random.uniform(0, 1 - a)
                arry_ori[i][j] = [a, b]
        arry = deepcopy(arry_ori)
        cout = 0
        lis = []
        for ii in list(G_copy):
            lis += [ii]
        random.shuffle(lis)
        for node in lis:
            for neibor in list(nx.neighbors(G_copy, node)):
                q_c_list = zij(node, neibor, x, omega_i, G_copy, arry)
                arry[node][neibor][0] = q_c_list[1]
                arry[node][neibor][1] = q_c_list[2]

        # G_copy = deepcopy(G)
        while len(max(nx.connected_components(G_copy), key=len)) > s:
            for i in range(0, T):
                lis = []
                for i in list(G_copy):
                    lis += [i]
                random.shuffle(lis)
                for node in lis:
                    for neibor in list(nx.neighbors(G_copy, node)):
                        q_c_list = zij(node, neibor, x, omega_i, G_copy, arry)
                        arry[node][neibor][0] = q_c_list[1]
                        arry[node][neibor][1] = q_c_list[2]
            q0_list = []
            for vertex in list(G_copy):
                q0_i = zi(vertex, x, omega_i, G_copy, arry)
                q0_list += [[vertex, q0_i]]
            q0_list.sort(key=lambda x: x[1])
            vertex_chosen = q0_list[-1][0]
            fvs += [vertex_chosen]
            G_copy.remove_node(vertex_chosen)
            cout += 1
        if len(fvs) < min_set_num:
            min_set_num = len(fvs)
            min_set = fvs
        print('x:', x, "len(fvs):", len(fvs), 'min_set:', min_set_num)
    print("min_set:", min_set)
    print(min_set_num)
    return min_set_num, min_set
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-n", type=int, default=50, help="number of vertices")
    parser.add_argument("-ave_degree", type=int, default=3.5, help="number of vertices")
    parser.add_argument("-C", type=int, default=5, help="C-dismantling")
    parser.add_argument("-type", type=str, default="ER", help="ER,RR,social_network")
    parser.add_argument("-name", type=str, default="911", help="karate,polbooks,football,lesmis,celegansneural,911")
    parser.add_argument("-seed", type=int, default=5, help="seed")
    parser.add_argument("-omega_i", type=int, default=1, help="parameter of BP")
    parser.add_argument("-T", type=int, default=15, help="iterations")
    parser.add_argument("-x_start", type=int,default=1, help="l_start")
    parser.add_argument("-x_step", type=float, default=0.1, help="step_of_x")
    parser.add_argument("-x_end", type=float,default=15, help="x_end")
    parser.add_argument("-BA_m", type=float, default=3, help="m in BA graph")
    args = parser.parse_args()
    min_set_num, min_set = main(args)


import networkx as nx
from copy import deepcopy
import time
import math
import random
import pandas as pd
import os
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

def zij(i, j, x, omega_i, g, arry):
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
    # print(z_i_j)
    q0_i_j = 1/z_i_j
    # print(q0_i_j)
    qi_i_j = (math.exp(x*omega_i))*fir/z_i_j
    # print(qi_i_j)
    return [z_i_j,q0_i_j,qi_i_j]

def zi(i, x, omega_i, g, arry):
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




# @jit
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
        for id in range(0,len(all_allocation)+1):
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
                if cut_number >= min_cut_number or len(sub_neighbors) >= cut_opt:

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

        print(counting, len(dismantling_set), dismantling_set)
        counting += 1
        min_cut_number = cut_number
        print(cut_number)
        min_dismantling_set = deepcopy(dismantling_set)



def copy_large_connected_components(graph, min_size):
    # 找到所有连通组件
    connected_components = list(nx.connected_components(graph))

    # 过滤出顶点数大于 min_size 的连通组件
    large_components = [comp for comp in connected_components if len(comp) > min_size]

    # 复制这些连通组件为新的图
    new_graphs = []
    for component in large_components:
        new_graph = graph.subgraph(component).copy()
        new_graphs.append(new_graph)

    return new_graphs


# # 创建一个示例图
# G = nx.erdos_renyi_graph(100, 0.05)
#
# # 复制顶点数大于 10 的连通组件
# min_size = 10
def main(args):
    omega_i = 1

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

    new_G = nx.Graph()
    for index, node in enumerate(G.nodes()):
        new_G.add_node(index)
    for edge in G.edges():
        source, target = edge
        new_source = list(G.nodes()).index(source)
        new_target = list(G.nodes()).index(target)
        new_G.add_edge(new_source, new_target)
    G = new_G
    arry_ori = []

    lis = []
    for i in range(0, args.n):
        lis += [i]
    random.shuffle(lis)


    min_set = []
    min_x = 0

    min_set_num = 100000000
    for x in range(1, args.step_end, args.step_length):
        fvs = []
        G_copy = deepcopy(G)
        for i in range(0, args.n):
            arry_ori += [[]]
            # print(arry)
            # print(nx.degree(G,i))
            for j in range(0, args.n):
                # print(arry)
                arry_ori[i] += [[0, 0]]
        # print(arry)
        for i in range(0, args.n):
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
        while len(max(nx.connected_components(G_copy), key=len)) > args.C:
            for i in range(0, args.T):
                lis = []
                for i in list(G_copy):
                    lis += [i]
                random.shuffle(lis)
                for node in lis:
                    for neibor in list(nx.neighbors(G_copy, node)):
                        q_c_list = zij(node, neibor, x, omega_i, G_copy, arry)
                        arry[node][neibor][0] = q_c_list[1]
                        arry[node][neibor][1] = q_c_list[2]
            # if cout == 0:
            #     for ver in list(G):
            #         print(zi(ver, x, omega_i,G_copy))
            max_q0 = 0
            q0_list = []

            for vertex in list(G_copy):
                q0_i = zi(vertex, x, omega_i, G_copy, arry)
                q0_list += [[vertex, q0_i]]
                # if q0_i > max_q0:
                #     max_q0 = q0_i
                #     q0_max_vertex = vertex
            q0_list.sort(key=lambda x: x[1])
            # print('max_q0',max_q0)
            # print(q0_list)
            vertex_chosen = q0_list[-1][0]
            fvs += [vertex_chosen]
            G_copy.remove_node(vertex_chosen)
            cout += 1
        if len(fvs) < min_set_num:
            min_set_num = len(fvs)
            min_set = fvs
            min_x = x
        largest_component = max(nx.connected_components(G_copy), key=len)
        print('x:' + str(x), 'len(fvs):' + str(len(fvs)))

    print('min_x', min_x)

    fvs = []
    G_copy = deepcopy(G)
    print(min_set_num, min_set)
    min_set_num = 100000000

    for i in range(0, args.n):
        arry_ori += [[]]
        # print(arry)
        # print(nx.degree(G,i))
        for j in range(0, args.n):
            # print(arry)
            arry_ori[i] += [[0, 0]]
    # print(arry)
    for i in range(0, args.n):
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
            q_c_list = zij(node, neibor, min_x, omega_i, G_copy, arry)
            arry[node][neibor][0] = q_c_list[1]
            arry[node][neibor][1] = q_c_list[2]

    # G_copy = deepcopy(G)
    while len(max(nx.connected_components(G_copy), key=len)) > args.BPD_C:
        for i in range(0, args.T):
            lis = []
            for i in list(G_copy):
                lis += [i]
            random.shuffle(lis)
            for node in lis:
                for neighbor in list(nx.neighbors(G_copy, node)):
                    q_c_list = zij(node, neighbor, min_x, omega_i, G_copy, arry)
                    arry[node][neighbor][0] = q_c_list[1]
                    arry[node][neighbor][1] = q_c_list[2]
        # if cout == 0:
        #     for ver in list(G):
        #         print(zi(ver, x, omega_i,G_copy))
        max_q0 = 0
        q0_list = []

        for vertex in list(G_copy):
            q0_i = zi(vertex, min_x, omega_i, G_copy, arry)
            q0_list += [[vertex, q0_i]]
            # if q0_i > max_q0:
            #     max_q0 = q0_i
            #     q0_max_vertex = vertex
        q0_list.sort(key=lambda x: x[1])
        # print('max_q0',max_q0)
        # print(q0_list)
        vertex_chosen = q0_list[-1][0]
        fvs += [vertex_chosen]
        G_copy.remove_node(vertex_chosen)
        cout += 1
    if len(fvs) < min_set_num:
        min_set_num = len(fvs)
        min_set = fvs
    print(min_x, len(fvs), min_set_num)

    G.remove_nodes_from(min_set)
    start = time.perf_counter()
    large_graphs = copy_large_connected_components(G, args.C)
    all = len(min_set) + args.num_of_pre_delete
    all_bpd = len(min_set) + args.num_of_pre_delete
    file_format = "graphml"
    for ijk, new_graph in enumerate(large_graphs):
        #     file_name = f"{ijk}.graphml"
        #     file_path = os.path.join('E:/PycharmProjects/pythonProject/venv\share/graph', file_name)
        #
        #     # 保存图文件
        #     nx.write_graphml(new_graph, file_path)
        #     print(f"Saved {file_path}")
        arry_ori = []

        lis = []
        for iij in range(0, args.n):
            lis += [iij]
        # print(lis)
        random.shuffle(lis)

        arry = deepcopy(arry_ori)
        cout = 0
        min_set = []
        min_set_num = 100000000
        for x in range(min_x, min_x + 1):
            fvs = []
            G_copy = deepcopy(new_graph)
            for ii in range(args.n):
                arry_ori += [[]]
                # print(arry)
                # print(nx.degree(G,i))
                for jj in range(args.n):
                    # print(arry)
                    arry_ori[ii] += [[0, 0]]
            # print(arry)
            for ii in list(G_copy):
                for jj in nx.neighbors(G_copy, ii):
                    a = random.uniform(0, 1)
                    b = random.uniform(0, 1 - a)
                    arry_ori[ii][jj] = [a, b]
            arry = deepcopy(arry_ori)
            cout = 0
            lis = []
            for ii in list(G_copy):
                lis += [ii]
            random.shuffle(lis)
            for node in lis:
                for neighbor in list(nx.neighbors(G_copy, node)):
                    q_c_list = zij(node, neighbor, x, omega_i, G_copy, arry)
                    arry[node][neighbor][0] = q_c_list[1]
                    arry[node][neighbor][1] = q_c_list[2]

            # G_copy = deepcopy(G)
            while len(max(nx.connected_components(G_copy), key=len)) > args.C:
                for i in range(0, args.T):
                    lis = []
                    for i in list(G_copy):
                        lis += [i]
                    random.shuffle(lis)
                    for node in lis:
                        for neighbor in list(nx.neighbors(G_copy, node)):
                            q_c_list = zij(node, neighbor, x, omega_i, G_copy, arry)
                            arry[node][neighbor][0] = q_c_list[1]
                            arry[node][neighbor][1] = q_c_list[2]
                # if cout == 0:
                #     for ver in list(G):
                #         print(zi(ver, x, omega_i,G_copy))
                max_q0 = 0
                q0_list = []

                for vertex in list(G_copy):
                    q0_i = zi(vertex, x, omega_i, G_copy)
                    q0_list += [[vertex, q0_i]]
                    # if q0_i > max_q0:
                    #     max_q0 = q0_i
                    #     q0_max_vertex = vertex
                q0_list.sort(key=lambda x: x[1])
                # print('max_q0',max_q0)
                # print(q0_list)
                vertex_chosen = q0_list[-1][0]
                fvs += [vertex_chosen]
                G_copy.remove_node(vertex_chosen)
                cout += 1
            if len(fvs) < min_set_num:
                min_set_num = len(fvs)
                min_set = fvs
            print(len(fvs), min_set_num)
        all_bpd += min_set_num
        min_dismantling_set = []
        counting = 0
        min_dismantling_set = []
        dismantling_set = []
        G_copy = deepcopy(G)
        cut_number = 0
        counting = 0
        cash = [[]]
        cash_num = [[]]
        cash_allo = [[]]
        num_of_choice = [0]
        cash_end = [0]
        dismantling_set = []
        min_cut_number = 20000
        n = len(new_graph)

        print(
            f"Graph {ijk}:  with {new_graph.number_of_nodes()} nodes and {new_graph.number_of_edges()} edges")

        branch_and_bound(new_graph, args.C, 0, [], cut_number, args.cut_opt)
        all += min_cut_number
        print(all)
        print(all_bpd)
        print('min_cut_number' + str(min_cut_number), 'dismantling_set' + str(min_dismantling_set))
        print(list(min_dismantling_set), str(min_dismantling_set))
        print('counting', counting)
        print('比例' + str((2 ** n) / counting))

    print("bd_based_bpd", all)
    print('bpd', all_bpd)
    end = time.perf_counter()
    return all, all_bpd, end-start


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-n", type=int, default=2000, help="number of vertices")
    parser.add_argument("-ave_degree", type=int, default=3, help="number of vertices")
    parser.add_argument("-C", type=int, default=5, help="C-dismantling")
    parser.add_argument("-type", type=str, default="ER", help="ER,RR")
    parser.add_argument("-seed", type=int, default=1, help="seed")
    parser.add_argument("-num_of_pre_delete", type=int, default=0, help="num_of_pre_delete")
    parser.add_argument("-cut_opt", type=int, default=100, help="cut the branch if neighbors of group greater than cut_opt")
    parser.add_argument("-BPD_C", type=int, default=100, help="C-dismantling problem solved by BPD")
    parser.add_argument("-step_length", type=int, default=1, help="step_length of x in BPD")
    parser.add_argument("-step_end", type=int, default=5, help="end of x in BPD")
    parser.add_argument("-T", type=int, default=20, help="number of loops in BPD")

    args = parser.parse_args()
    bd_based_bpd, bpd, time = main(args)




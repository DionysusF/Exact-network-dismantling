from bd_random_social import main
import argparse
from openpyxl import Workbook

# 创建 Namespace 对象模拟 argparse 的结果
wb = Workbook()
ws = wb.active
ws.append(["n", "seed", "time", "k", "min_cut_number", "min_dismanting_set"])  # 表头

for nn in range(50,72,2):
    for seedseed in range(1,2):
        # 调用程序A的函数
        args = argparse.Namespace(type='RR', n=nn, ave_degree=3, C=3, seed=seedseed, num_of_pre_delete=0, cut_opt=100, mode="size_only")
        time, k, min_cut_number, min_dismantling_set = main(args)  # 解包返回值

        # 写入Excel
        ws.append([
            nn,  # n
            seedseed,  # seed
            time,  # time
            k,  # k
            min_cut_number,  # min_cut_number
            str(min_dismantling_set)  # 将集合转为字符串存储
        ])
        print(f"n={nn}, seed={seedseed} -> time={time}, k={k}")

# 保存Excel文件
wb.save("output_RR.xlsx")
print("结果已保存到 output_RR.xlsx")
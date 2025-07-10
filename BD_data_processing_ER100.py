from bd_random_social import main
import argparse
from openpyxl import Workbook

# 创建 Namespace 对象模拟 argparse 的结果
wb = Workbook()
ws = wb.active
ws.append(["参数组", "n", "seed", "time", "k", "min_cut_number", "min_dismanting_set"])  # 表头

for nn in [100]:
    for seedseed in (926548, 265982, 354821, 652841,146257):
        # 调用程序A的函数
        args = argparse.Namespace(type='ER', n=nn, ave_degree=3.5, C=5, seed=seedseed, num_of_pre_delete=0, cut_opt=5, mode="size_only")
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
wb.save("output_ER_100.xlsx")
print("结果已保存到 output_ER_100.xlsx")
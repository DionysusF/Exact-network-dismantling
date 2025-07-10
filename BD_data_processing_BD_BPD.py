import argparse
from openpyxl import Workbook
from BPD_based_BD import main
# 创建 Namespace 对象模拟 argparse 的结果
wb = Workbook()
ws = wb.active
ws.append(["n", "seed", "time", "BPD-BD", "min_cut_number", ])  # 表头

for nn in [100]:
    for seedseed in range(1,6):
        # 调用程序A的函数
        args = argparse.Namespace(type='ER', n=nn, ave_degree=3.5, C=5, seed=seedseed, BPD_C=100, num_of_pre_delete=0, cut_opt=100, T=15, step_end=15, step_length=1)
        BD, BPD, time = main(args)  # 解包返回值

        # 写入Excel
        ws.append([
            nn,
            seedseed,
            time,
            BPD-BD,
            BPD,
        ])
        print(f"n={nn}, seed={seedseed} -> time={time}, k={k}")

# 保存Excel文件
wb.save("output_ER_100.xlsx")
print("结果已保存到 output_ER_100.xlsx")

#吴恩达线性回归模型
import numpy as np
import matplotlib.pyplot as plt
#从txt文件中导入数据
def load_data(filename):
    file = open(filename)
    data=[]
    for line in file.readlines():
        line_split = line.strip().split(",")
        col_num=len(line_split)
        temp=[]
        for i in range(col_num):
            temp.append(float(line_split[i]))
        data.append(temp)
    return np.array(data)
data = load_data('ex1data1.txt')
print(data.shape)
print(data)
print(type(data))
print(data[::])
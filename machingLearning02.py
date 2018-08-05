import numpy as np
import matplotlib.pyplot as plt
#线性回归模型
#首先从txt文件中导入数据
def load_data(filename):
    file=open(filename)
    data=[]
    for line in file.readlines():
        lineArr = line.strip().split(",")
        clo_num = len(lineArr)
        temp=[]
        for i in range(clo_num):
            temp.append(float(lineArr[i]))
        data.append(temp)
    return np.array(data)

data=load_data('ex2data1.txt')
print(data.shape)
print(data[:5])
X=data[:,:-1]
Y=data[:,-1:]
print(X.shape)
print(Y.shape)
print(X[:5])
print('-----------------------')
print(Y[:5])




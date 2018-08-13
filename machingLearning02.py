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
X=data[:,:-1]
print('~~~~~~~~~~~~~~~~~~~~')
print(X)
print('-------------------')
Y=data[:,-1:]
print(Y)
print('~~~~~~~~~~~~~~~~~~~~')
print(Y.ravel())
label0=np.where(Y.ravel()==0)
print(label0)
plt.scatter(X[label0,0],X[label0,1],marker='x',color='r',label='No admitted')
label1=np.where(Y.ravel()==1)
plt.scatter(X[label1,0],X[label1,1],marker='o',color = 'b',label = 'Admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(loc='upper left')
plt.show()
#定义一个sigmod函数
def sigmod(x):
    return 1/(1+np.exp(-x))
#定义矩阵的乘积
def out(x,w):
    return sigmod(np.dot(x,w))
#定义损失函数
def compute_cost(X_train,y_train,theat):
    m=X_train.shape[0]
    J=0
    theat=theat.reshape(-1,1)
    grad=np.zeros((X_train.shape[1],1))
    h=out(X_train,theat)
    J=-1*np.sum(y_train*np.log(h)+(1-y_train)*np.log((1-h)))/m
    grad=X_train.T.dot((h-y_train))/m
    grad=grad.ravel()
    return J, grad
#代码测试
m=X.shape[0]
one=np.ones((m,1))
print(one)
X=np.hstack((one,data[:,:-1]))
print(X)
W=np.zeros((X.shape[1],1))
cost,grad=compute_cost(X,Y,W)
print('compute with w=[0,0,0]')
print('Excepted cost : 0.693')
print(cost)
print('Excepted gradients:[-0.1,-12,-11]')
print(grad)


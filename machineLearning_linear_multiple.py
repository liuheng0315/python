#多元线性回归
import numpy as np
import matplotlib.pyplot as plt
#导入数据
def load_data(filename):
    data=[]
    file=open(filename)
    for line in file.readlines():
        lineArr=line.strip().split(',')
        col_num=len(lineArr)
        temp=[]
        for i in range(col_num):
            temp.append(int(lineArr[i]))
        data.append(temp)
    return np.array(data)
data=load_data('ex1data2.txt')
X=data[:,:-1]
y=data[:,-1:]
print(X.shape)
print("X:",X)
print(y.shape)
print("y:",y)

#定义特征缩放(均值归一化)
def featureNormalize(X):
    X_norm=X
    mu=np.zeros((1,X.shape[1]))
    sigma=np.zeros((1,X.shape[1]))
    mu=np.mean(X,axis=0)
    sigma=np.std(X,axis=0)
    X_norm = (X - mu) / sigma
    return X_norm,mu,sigma

X_norm,mu,sigma=featureNormalize(data[:,:-1])
num_train=X.shape[0]
one=np.zeros((num_train,1))
X=np.hstack((one,X_norm))
W=np.zeros((X.shape[1],1))#初始化全为零

#计算其损失函数
def compute_cost(X_test,y_test,theta):
    num_X=X_test.shape[0]
    cost=0.5*np.sum(np.square(X_test.dot(theta)-y_test))/num_X
    return cost


#计算梯度下降
def gradient_descent(X_test,y_test,theta,alpha=0.005,iters=1500):
    J_history=[]
    num_X=X_test.shape[0]
    for i in range(iters):
        theta=theta-alpha * X_test.T.dot((X_test.dot(theta)-y_test))/num_X
        cost=compute_cost(X_test,y_test,theta)
        J_history.append(cost)
    return theta,J_history

#测试结果
theta,J_history=gradient_descent(X,y,W)
print("----------------------------")
print('Theta computed from gradient descent: ',theta)
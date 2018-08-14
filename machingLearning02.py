import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
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
    print('grad1111111',grad)
    grad=grad.ravel()
    print('grad222222',grad)
    return J,grad
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


#测试2
cost1,grad1=compute_cost(X,Y,np.array(([[-24],[0.2],[0.2]])))
print ('compute with w=[-24,0.2,0.2]')
print ('Expected cost (approx):0.218....')
print (cost1)
print ('Expected gradients (approx): [0.04,2.566,0.646]')
print (grad1)


#选择了最优化算法，并非是梯度下降算法
params=np.zeros((X.shape[1],1)).ravel()
agrs=(X,Y)
def f(params,*args):
    X_train,y_train=args
    m,n=X_train.shape
    J=0
    theta=params.reshape((n,1))
    h=out(X_train,theta)
    J=-1*np.sum(y_train*np.log(h)+(1-y_train)*np.log((1-h)))/m
    print('JJJJJ',J)
    return J

def gradf(params,*args):
    X_train,y_train=args
    m,n=X_train.shape
    theta=params.reshape(-1,1)
    h=out(X_train,theta)
    grad=np.zeros((X_train.shape[1],1))
    grad=X_train.T.dot((h-y_train))/m
    g=grad.ravel()
    return g
res=optimize.fmin_cg(f,x0=params,fprime=gradf,args=agrs,maxiter=500)
print(res)

#可视化线性的决策边界
label=np.array(Y)
index_0=np.where(label.ravel()==0)
plt.scatter(X[index_0,1],X[index_0,2],marker='x'\
            ,color = 'b',label = 'Not admitted',s = 15)
index_1 =np.where(label.ravel()==1)
plt.scatter(X[index_1,1],X[index_1,2],marker='o',\
            color = 'r',label = 'Admitted',s = 15)
#展示决策边界
x1=np.arange(20,100,0.5)
x2=(-res[0]-res[1]*x1)/res[2]
plt.plot(x1,x2,color='black')
plt.xlabel('x1')
plt.ylabel('y1')
plt.legend(loc='upper left')
plt.show()

#预测函数
def predict(X,theta):
    h=out(X,theta)
    y_pred=np.where(h>=0.5,1.0,0)
    return y_pred
prob=out(np.array([[1,45,85]]),res)
print(prob)
p=predict(X,res)
print(np.mean(p==Y.ravel()))
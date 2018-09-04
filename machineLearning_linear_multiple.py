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
one=np.ones((num_train,1))
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
        theta=theta-alpha * X_test.T.dot(X_test.dot(theta)-y_test)/num_X
        cost=compute_cost(X_test,y_test,theta)
        J_history.append(cost)
    return theta,J_history

#测试结果
theta,J_history=gradient_descent(X,y,W)
print("----------------------------")
print('Theta computed from gradient descent: ',theta)

plt.plot(J_history,color = 'b')
plt.xlabel('iters')
plt.ylabel('j_cost')
plt.title('cost variety')
plt.show()

#测试，需要使用训练集的均值和方差区进行特征缩放
X_t=([[1650,3]]-mu)/sigma
X_test=np.hstack((np.ones((1,1)),X_t))
predict=X_test.dot(theta)
print ('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent)')
print (predict)


#不使用梯度下降法，直接使用公式求theta
XX=data[:,:-1]
yy=data[:,-1:]
m=XX.shape[0]
one = np.ones((m,1))
XX=np.hstack((one,XX))
print(XX.shape)
print(yy.shape)

def normalEquation(X_train,y_train):
    w=np.zeros((X_train.shape[1],1))
    w=((np.linalg.pinv(X_train.T.dot(X_train))).dot(X_train.T)).dot(y_train)
    return w

w=normalEquation(XX,yy)
print ('Theta computed from the normal equations:',w)

price = np.dot(np.array([[1,1650,3]]),w)
print ('Predicted price of a 1650 sq-ft, 3 br house (using normal equations)',price)

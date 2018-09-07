#本文是onevsall多分类器的实现
#导入手写字体的数据集
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt

digits=load_digits()
print(digits.keys())
data=digits.data
target=digits.target
print(data.shape)
print(target.shape)
print('the image 15 is',target[15])

#随机选择50个数据
classes=['0','1','2','3','4','5','6','7','8','9']
num_classes=len(classes)
samples_per_class=5
for y,cla in enumerate(classes):
    idxs=np.flatnonzero(target==y)
    idxs=np.random.choice(idxs,samples_per_class,replace=False)#choice为随机采样，idsx为采样数组
    for i,idx in enumerate(idxs):
        plt_idx=i*num_classes+y+1
        plt.subplot(samples_per_class,num_classes,plt_idx)
        plt.imshow(digits.images[idx].astype('uint8'))
        plt.axis('off')
        if i==0:
            plt.title(cla)
plt.show()

#正则化逻辑回归算法
def sigmoid(x):
    return 1/(1+np.exp(-x))

def out(x,w):
    return sigmoid(np.dot(x,w))

def f(param,*args):
    X_train,y_train,reg=args
    m,n=X_train.shape
    J=0
    theta=param.reshape((n,1))
    h=out(X_train,theta)
    theta_1=theta[1:,:]
    J=-1*np.sum(y_train*np.log(h)+(1-y_train)*np.log(1-h))/m+reg*(theta_1.T.dot(theta_1))/(2*m)
    return J

def gradf(params,*args):
    X_train,y_train,reg=args
    m,n=X_train.shape
    theta=params.reshape(-1,1)
    h=out(X_train,theta)
    grad=np.zeros((X_train.shape[1],1))
    theta_1=theta[1:,:]
    grad=X_train.T.dot((h-y_train))/m
    grad[1:,:]+=reg*theta_1/m
    g=grad.ravel()
    return g


# 吴恩达线性回归模型
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 从txt文件中导入数据
def load_data(filename):
    file = open(filename)
    data = []
    for line in file.readlines():
        line_split = line.strip().split(",")
        col_num = len(line_split)
        temp = []
        for i in range(col_num):
            temp.append(float(line_split[i]))
        data.append(temp)
    return np.array(data)


data = load_data('ex1data1.txt')
print(data.shape)
x = data[:, :-1]
y = data[:, -1:]
# 可视化数据
plt.scatter(x, y, color='r', marker='x')
plt.xlabel('X')
plt.ylabel('Y')
# plt.show()
# 计算损失函数
num_train = x.shape[0]
# print(num_train)
one = np.ones((num_train, 1))
# print(one.shape)
x = np.hstack((one, data[:, :-1]))
print(x[:])
w = np.zeros((2, 1))
print(w)


# 定义计算的cost函数
def comput_cost(x_test, y_test, theta):
    num_x = x_test.shape[0]
    cost = 0.5 * np.sum(np.square(x_test.dot(theta) - y_test)) / num_x
    return cost


cost_1 = comput_cost(x, y, w)
print('cost=%f,with w=[0,0]' % cost_1)
cost_2 = comput_cost(x, y, np.array([[-1], [2]]))
print('cost=%f,with w=[-1,2]' % cost_2)


# print(np.array([[-1], [2]]))
# print(type(np.array([[-1], [2]])))
# 定义梯度下降函数，更新参数theta
def gradient_descent(x_test, y_test, theta, alpha=0.01, iters=1500):
    J_history = []
    num_x = x_test.shape[0]
    for i in range(iters):
        theta = theta - alpha * x_test.T.dot((x_test.dot(theta) - y_test)) / num_x
        cost = comput_cost(x_test, y_test, theta)
        J_history.append(cost)
    return theta, J_history


theta, J_history = gradient_descent(x, y, np.array([[0.001], [0.001]]))
print(theta)
print(theta.shape)
print(J_history[-1])
print("---------------")
# 使用计算出来的theta进行预测
predict1 = np.array([[1, 3.5]]).dot(theta)
predict2 = np.array([[1, 7]]).dot(theta)
print(predict1 * 10000, predict2 * 10000)
# 可视化回归曲线
plt.subplot(211)
plt.scatter(x[:, 1], y, color='r', marker='x')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(x[:, 1], x.dot(theta), '-', color='black')
# 可视化close曲线
plt.subplot(212)
plt.plot(J_history)
plt.xlabel('iters')
plt.ylabel('cost')
plt.show()
# 可视化3D图形，因为J与theta1和theta0两个参数相关
size=100
theta0Vals = np.linspace(-10, 10, size)
theta1Vals = np.linspace(-1, 4, size)
JVals=np.zeros((size,size))
for i in range(size):
    for j in range(size):
        col=np.array([[theta0Vals[i]],[theta1Vals[j]]]).reshape(-1,1)
        JVals[i,j]=comput_cost(x,y,col)

theta0Vals,theta1Vals=np.meshgrid(theta0Vals,theta1Vals)
JVals=JVals.T
print(JVals.shape,JVals[0,0],JVals[1,1])
fig=plt.figure()
ax=Axes3D(fig)
ax.plot_surface(theta0Vals,theta1Vals,JVals)
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.set_zlabel(r'$J(\theta)$')
plt.show()
#绘制曲线轮廓
contourFig=plt.figure()
ax = contourFig.add_subplot(111)
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
CS=ax.contour(theta0Vals,theta1Vals,JVals,np.logspace(-2,3,20))
plt.clabel(CS, inline=1, fontsize=10)
# 绘制最优解
ax.plot(theta[0,0],theta[1,0],'rx',markersize=10, linewidth=2)
plt.show()


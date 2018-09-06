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
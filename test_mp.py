# import numpy as np
# import matplotlib.pyplot as plt 

# fig = plt.figure(2)
# x = range(10)
# y = [3, 4,5 ,3, 6, 8, 4, 3, 9, 10]
# plt.plot(x, y)
# plt.show()

# fig = plt.figure()
# x = np.array(range(10))
# y = x* x + 1
# plt.plot(x, y)
# plt.show()

# fig = plt.figure()
# x = np.linspace(-1, 1, 50)
# y = 2*x + 1
# plt.plot(x, y, 'bo')
# plt.show()

# fig = plt.figure(1)
# x = np.arange(0., 5., 0.2)
# plt.plot(x, x, 'r--')
# plt.plot(x, x**2, 'bs')
# plt.plot(x, x**3, 'g^')
# plt.show()

# fig = plt.figure(2)
# x = np.array(range(10))
# y = 2*x + 1
# plt.plot(x, y)
# plt.show()

# import numpy as np  
# import matplotlib.pyplot as plt
# x = np.linspace(-1, 1, 50)
# fig = plt.figure(figsize=(10, 8), dpi = 100)
# plt.plot(x, x**2)
# plt.xlabel('x axis', loc = 'left', color = 'red')
# # plt.savefig('test_mp.png')
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
# x = np.linspace(-1, 1, 40)
# y = np.linspace(30, 50, 40)
# fig = plt.figure()
# plt.plot(x, y)
# plt.xticks(np.linspace(-1, 1, 10))
# plt.yticks(np.linspace(25, 45, 10))
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt  
# x = range(10)
# y = [15,14,26,26,15,18,19,8,21,22]
# plt.plot(x, y)
# plt.xticks(range(10))
# plt.yticks(range(min(y),max(y)+1,2))  
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt  
# x = [1, 2, 3, 4]
# y = [1, 4, 9, 6]
# labels = ['Frogs', 'Tiger', 'Pig', 'Sheep']
# fig = plt.figure()
# plt.plot(x, y)
# plt.xticks(x, labels, rotation='45')
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# x = np.linspace(0, 10, 1000)
# y = np.sin(x)
# z = np.cos(x**2)
# plt.figure()
# plt.plot(x, y, label = '$ sin(x) $', color = 'red')
# plt.plot(x, z, 'b--',label = '$ cos(x^2)$')
# plt.xlabel('Times(s)')
# plt.ylabel('Volt')
# plt.legend(loc = 'upper right' )
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# x = np.linspace(-1, 1, 50)
# plt.figure()
# plt.plot(x, x**2, 'r--', label = '$ x^2$')
# plt.legend(loc = 'upper center')
# plt.text(0. , 0.6, 'curve', fontsize= 20, color = 'blue', rotation = 45)
# plt.show()


# import numpy as np  
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# x = np.linspace (-1,1,50)
# y = x**2 
# fig = plt.figure()
# plt.plot(x, y)
# plt.text(0.,0.6,'二次曲线')
# plt.show()

# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
# import numpy as np 

# img = mpimg.imread('test_mp.png')
# plt.imshow(img)
# # plt.axis('off')
# plt.show()

# import numpy as np  
# import matplotlib.pyplot as plt
# np.random.seed(1)
# n = 50
# x = np.random.rand(n)
# y = np.random.rand(n)
# colors = np.random.rand(n)
# area = (30 * np.random.rand(n))**2  
# plt.scatter(x, y, s=area, c=colors, alpha=0.5)
# plt.show()

# import numpy as np  
# import matplotlib.pyplot as plt
# x_3 = range(30)
# x_10 = range(40,70)
# y_3 = [11,17,16,11,12,11,7,6,7,8,9,12,15,14,17,18,21,16,17,20,14,15,15,15,19,21,22,22,22,23]
# y_10 = [26,26,28,19,21,17,16,19,18,20,20,19,22,23,17,20,21,22,15,11,15,7,13,17,10,11,13,12,13,8]
# plt.scatter(x_3, y_3)
# plt.scatter(x_10, y_10)
# plt.show()

# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# a = ['战狼2','长津湖','月球陨落','年少有你','心想念动','神秘海域'] # 电影名
# b = [60,70,30,25,20,35] # 票房
# plt.barh(a, b, 0.35, color = 'green')
# plt.show()

# import matplotlib.pyplot as plt
# labels = ['G1', 'G2', 'G3', 'G4', 'G5']
# men_means = [20, 35, 30, 35, 27]
# women_means = [25, 32, 34, 20, 25]
# men_std = [2, 3, 4, 1, 2]
# women_std = [3, 5, 2, 3, 3]
# width = 0.35       
# plt.bar(labels, men_means, width, yerr=men_std, label='Men')
# plt.bar(labels, women_means, width, yerr=women_std, bottom=men_means,label='Women')
# plt.ylabel('Scores')
# plt.title('Scores by group and gender')
# plt.legend()
# plt.show()

# import numpy as np
# from matplotlib import pyplot as plt
# lin_data = np.arange(1,9)
# q_data = lin_data ** 2
# plt.figure()
# x_value = list (range(len(lin_data)))
# plt.bar (x_value,lin_data,width = 0.3)
# plt.bar (x_value,q_data,width = 0.3,bottom = lin_data)
# plt.show()

# import matplotlib.pyplot as plt
# data = [20,30,33,7,76,99,31,57,33,74,90,2,15,11,0,41,13,43,6]
# bin_width = 9
# bins = (max(data) - min(data))//bin_width
# # plt.xlim(0, 99)
# plt.xticks(range(0, 100, 9))
# plt.hist(data, bins)
# plt.show()

# import numpy as np  
# import matplotlib.pyplot as plt
# mu, sigma = 100, 15
# x = mu + sigma * np.random.randn(10000) 
# plt.hist(x, 50, density=True, facecolor='g', alpha=0.75)
# plt.xlabel('Smarts')
# plt.ylabel('Probability')
# plt.title('Histogram of IQ')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.xlim(40, 160)
# plt.ylim(0, 0.03)
# plt.grid(True)
# plt.show()

# import numpy as np  
# import matplotlib.pyplot as plt
# labels=['China','Swiss','USA','UK','Laos','Spain']
# X=[222,42,455,664,454,334]

# fig = plt.figure()
# plt.pie(X, labels = labels,autopct = '%.2f%%')
# plt.title('Pie Chart')
# plt.show()


# import matplotlib.pyplot as plt
# # 定义标签
# labels = ['Frogs', 'Tigers', 'Dogs', 'Pigs'] 
# # 每一块的比例
# sizes = [15, 30, 45, 10]
# # 每一块的颜色
# colors = ['yellow', 'gold', 'cyan', 'red']
# # 突出显示，这里仅仅突出显示第二块（即'Tigers'）
# explode = (0, 0.1, 0, 0)  
# plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
# plt.show()


# # 使用plt.subplots() 函数
# import matplotlib.pyplot as plt
# import numpy as np
# fig, ax = plt.subplots(1, 2, figsize= (12, 5))
# x = np.random.rand(100)
# y = np.random.rand(100)
# categories = ['A', 'B', 'C', 'D']
# values = [23, 45, 56, 78]
# #第一个图中绘制散点图
# ax[0].scatter(x, y, color = 'blue', alpha = 0.75)
# ax[0].set_title('Scatter Plot')
# ax[0].set_xlabel('x')
# ax[0].set_ylabel('y')
# # 第二个子图中绘制柱形图
# ax[1].bar(categories, values, color = 'red')
# ax[1].set_title('Bar Chart')
# ax[1].set_xlabel('Categories')
# ax[1].set_ylabel('Values')

# plt.show()

# import numpy as np  
# import matplotlib.pyplot as plt
# t = np.arange(0.0, 2.0, 0.01)
# s = 1 + np.sin(2 * np.pi * t)
# fig, ax = plt.subplots()
# ax.plot(t, s)
# ax.set(xlabel='time (s)', ylabel='voltage (mV)',
#        title='About as simple as it gets')
# ax.grid()
# fig.savefig("test.png")
# plt.show()


# import numpy as np  
# import matplotlib.pyplot as plt
# x1 = np.linspace(0.0, 5.0)
# x2 = np.linspace(0.0, 2.0)
# y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
# y2 = np.cos(2 * np.pi * x2)
# fig, (ax1, ax2) = plt.subplots(2, 1)
# fig.suptitle('A tale of 2 subplots')
# ax1.plot(x1, y1, 'o-')
# ax1.set_ylabel('Damped oscillation')
# ax2.plot(x2, y2, '.-')
# ax2.set_xlabel('time (s)')
# ax2.set_ylabel('Undamped')
# plt.show()


import numpy as np  
import matplotlib.pyplot as plt
fig = plt.figure()
fig.subplots_adjust(top=0.8)
ax1 = fig.add_subplot(211)
ax1.set_ylabel('volts')
ax1.set_title('a sine wave')

t = np.arange(0.0, 1.0, 0.01)
s = np.sin(2*np.pi*t)
line = ax1.plot(t, s, color='blue', lw = 5)
plt.show()
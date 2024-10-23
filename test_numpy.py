import numpy as np 

# a = np.arange(1, 2, 0.2)

# print(a)

# a = np.linspace(1, 2, 10, endpoint = False)
# print(a)

# a = np.logspace(2.0, 3.0)
# print(a)

# print(np.ones((4, 4)))
# print(np.eye(4))
# print(np.eye(3, 4))
# print(np.eye(4, k = 1))

# a = np.zeros(10)
# # print(a)
# b = np.ones((3, 4))
# # print(b)
# c = np.arange(1, 11)
# # print(c)
# d = np.arange(0, 9)
# d = np.reshape(d, (3, 3))
# print(d)

# a = np.array([[1, 2], [3, 4]])
# ave = np.average(a)
# print(ave)

# a = np.arange(10)
# # np.random.shuffle(a)
# print(a)
# b = np.random.choice(a, size = 5)
# print(b)

# a = np.array([2, 2, 3, 4, 4, 10, 1, 3])
# b = np.unique(a)
# print(b)

# s = [590,585,570,585,570,570] # 总成绩
# m = [90,  85, 60, 68, 65, 70] # 数学成绩
# # Sort by sum, then by math
# rank = np.lexsort((m,s))
# print(rank)

# a = np.loadtxt('test.txt', usecols = [0, 3])
# print(a)

# arr = np.array([1, 5, 2, 3, 6, 8, 7, 9])
# np.savetxt('ar.txt', arr, fmt = '%d', delimiter = ' ')
# f = np.loadtxt('ar.txt')
# print(f)
# arr1 = np.array([[1, 2,3, 4],[5, 6,7,8]])

# for i in arr1.flat:
#     print(i)
# import numpy as np
# x = np.array([1.4523,2.7348,3.1652])
# x = np.around(x,2)
# print(x)

# y = np.array([1.11111, 3.2252, 6.45444])
# y1 = np.ceil(y)
# y2 = np.floor(y)

# print(y1)
# print(y2)

# a = np.array([1,2 ,3,4])
# b = np.array([6, 7, 8, 9])
# y = np.vstack((a, b))
# x = np.hstack((a, b))
# print(y)
# print(x)

# a = np.arange(1, 13).reshape(3, 4)
# print('before: ', a)
# a[:,[0,2]] = a[:,[2,0]]
# print('after:', a)

# arr = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
# # x = arr[:,0:3]
# # x = arr[1]
# arr[[0,1],:] = arr[[1,0],:]
# print(arr)

# arr = np.random.rand(10, 10)
# x = arr.max(axis = 0)
# y = arr.min()
# print(x)
# print(y)

# arr = np.random.rand(10,)
# arr[arr.argmax()] = 1
# print(arr.argmax())
# print(arr)

# row = np.arange(0, 5)
# matrix = np.tile(row, (5, 1))
# print(matrix)

# row = np.random.randint(1, 11, (5, 5))
# print(row)

# arr = np.arange(0, 5)
# mat = np.tile(arr, (5, 1))
# print(mat)

# arr = np.random.rand(10, 10)

# print(arr.std())

# a = np.array([1, 2, 3, 4, 5])

# res = np.zeros((3, 3))

# res[-1,:] = a[-3:]
# print(res)

# list1 = eval(input())
# list2 = eval(input())

# arr1 = np.array(list1)
# arr2 = np.array(list2)

# print(arr1.dot(arr2))

# arr1 = np.array([[1, 2, 3], [4, 5, 6]])
# arr2 = np.array([[1, 2, 3], [4,5, 7]])

# for i in arr1.flat:
#     if i not in arr2:
#         print(i)

# arr = np.arange(100)
# subarrays = [arr[i:i+10] for i in range(0, 100, 10)]

# for i, subarray in enumerate(subarrays):
#     print(f'subarrays {i+1}: {subarray}')

# arr = np.random.rand(5, 4)
# print(arr)
# print(np.sum(arr, axis = 0))

def Rank(arr):
    element = [(x, i) for i, x in enumerate(arr)]
    sorted_elements = sorted(element, key = lambda x : x[0])

    rank_dict = {index: rank for rank, (_, index) in enumerate(sorted_elements)}
    rank_arr = [rank_dict[i] for i in range(len(arr))]
    return rank_arr
arr = [3, 1, 4, 1, 5, 9, 2,6]
x = Rank(arr)
print(x)


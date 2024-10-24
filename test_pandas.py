# import pandas as pd
# b = pd.Series([1,5,3,4,10,0,9])
# print (b.values)
# print (b.index)
# print (type(b.values))

# import pandas as pd
# s = pd.Series ([21,19,20,50], index = ['张三','李四','王五','赵六'])
# print (s)

# import pandas as pd
# country_dicts = {'CH': '中国', 'US': '美国', 'AU': '澳洲'}
# country_dict_s = pd.Series(country_dicts)
# print(country_dict_s)

# import pandas as pd
# data = [1,2,3,4,5]
# ind = ['a','b','c','d','e']
# s = pd.Series (data, index = ind )
# # print (s)
# # print (s.loc[['a','b','e']])
# print (s.iloc[[0,1,4]])

# import pandas as pd
# s = pd.Series ([21,19,20,50], index = ['张三','李四','王五','赵六'])
# s1 = s.to_dict ()
# print (s1)

# import pandas as pd
# df = pd.DataFrame()
# print(df)

#通过 ndarray 构建 DataFrame
# import pandas as pd
# import numpy as np
# arr = np.arange(20).reshape(5,4)
# r = range(1,6)
# c = ['A','B','C','D']
# df = pd.DataFrame(arr,index = r,columns = c)
# print(df)

# 通过 list 创建Dataframe
# import pandas as pd
# lst = [[11,24,53],
#       [14,51,63],
#       [72,83,69]]
# df = pd.DataFrame(lst)
# print(df) 

# 通过 Series 创建Dataframe
# import pandas as pd
# country1 = pd.Series({'Name': '中国','Language': 'Chinese','Area': '9.597M km2','population': 14})
# country2 = pd.Series({'Name': '美国','Language': 'English','Area': '9.834M km2','population': 3})
# country3 = pd.Series({'Name': '澳洲','Language': 'English', 'Area':'7.692M km2','population': 58})
# df = pd.DataFrame([country1, country2, country3], index=['CH', 'US', 'AU'])
# print(df)

# 通过 dict 创建Dataframe
# import pandas as pd
# dic = {'c1':[1,2,3],
#        'c2':[4,5,6],
#        'c3':[7,8,9]}
# df = pd.DataFrame(dic)
# print(df)

# 通过 list 和 dict 创建Dataframe
# import pandas as pd
# names = ['张三','李四','王五']
# ages = [23,45,12]
# df = pd.DataFrame({'姓名':names,'年龄':ages})
# print(df)

# import pandas as pd
# df = pd.read_csv('4.csv')
# print(df)

# import pandas as pd
# data = pd.read_excel('test.xlsx')
# print('显示表格的属性:',data.shape)
# print('显示表格的列名:',data.columns)
# print('显示表格前50行:',data.head(50))

# import pandas as pd
# df =pd.DataFrame ([[1,2,3],[4,5,6],[7,8,9]],
#                    index = ['A','B','C'],
#                    columns = ['d1','d2','d3'])
# print (df)

# import pandas as pd
# country1 = pd.Series({'Name': '中国','Language': 'Chinese','Area': '9.597M km2','Happiness Rank': 79})
# country2 = pd.Series({'Name': '美国','Language': 'English (US)','Area': '9.834M km2','Happiness Rank': 14})
# df = pd.DataFrame([country1, country2], index=['CH', 'US'])
# df['Location'] = '地球'
# print(df)

# import pandas as pd
# df =pd.DataFrame ([[1,2,3],[4,5,6],[7,8,9]],
#                    index = ['A','B','C'],
#                    columns = ['d1','d2','d3'])
# print (df)

# del df['d2']
# print(df)

# import pandas as pd
# df = {'购药时间':['2021-04-1 周四','2021-04-3 周六','2021-04-05 周一'],
#              '社保卡号':['0012616528','0012616534','0012602828'],
#              '商品编号':[236701,236704,236712],
#              '商品名称':['板蓝根','VC银翘片','泰诺口服液'],
#              '销售数量':[6,10,3],
#              '应收金额':[20.3,36.8,73.2]
# }
# df = pd.DataFrame (df)
# print(df)
# # print(df.iloc[1,:])
# # print(df.loc[:,'商品名称'])
# # print(df[['商品名称', '销售数量']])
# # print(df.loc[1:2, '社保卡号':'销售数量'])
# query = df.loc[:, '销售数量'] > 3
# print(df.loc[query,:])

# import pandas as pd
# df =pd.DataFrame ([[1,2,3],[4,5,6],[7,8,9]],
#                    index = ['A','B','C'],
#                    columns = ['d1','d2','d3'])
# print (df)
# # print(df.describe())
# print(df.info())

# import pandas as pd
# df = pd.DataFrame ({'id': ['2021308', '2021318','2021305','2021303'],
#                     '数学':[91, 88, 75, 68],'物理': [81, 82, 87, 76],
#                      'python': [94, 81, 86, 71]},index = [1,2,3,4])
# print(df)
# print(df.drop([1,4]))

# import pandas as pd
# df = {'购药时间':['2021-04-1 周四','2021-04-3 周六','2021-04-05 周一'],
#              '社保卡号':['0012616528','0012616534','0012602828'],
#              '商品编号':[236701,236704,236712],
#              '商品名称':['板蓝根','VC银翘片','泰诺口服液'],
#              '销售数量':[6,10,3],
#              '应收金额':[20.3,36.8,73.2]
# }
# df = pd.DataFrame (df)
# print(df.dtypes)
# df['商品编号'] = df['商品编号'].astype(str) 
# df['销售数量'] = df['销售数量'].astype('float32')
# print(df.dtypes)

# import pandas as pd
# data = {'name':['Joe','Cat','Mike','Kim','Amy'],'year':[2012,2012,2013,2018,2018],'Points':[4,25,6,2,3]}
# df = pd.DataFrame (data, index = ['Day1','Day2','Day3','Day4','Day5'])
# print (df)
# print(df['year'].unique())


## 这里跑不过去
# import pandas as pd
# df = pd.DataFrame ({'key1': ['a','a','b','b','a'],
#                            'key2': ['one','two','one','two','one'],
#                            'data1': [2,5,6,7,9],
#                            'data2': [8,1,5,2,3]})
# # print(df)
# del df['key2']

# g = df.groupby(df['key1'])
# # print(g.sum())
# print(g.mean())


# import numpy as np
# import pandas as pd
# x = np.random.randint(10, size=7)
# y = np.random.randint(10, size=7)
# years = np.arange(2015,2022)
# keys1 = ['A','B','A','B','A','C','C']
# keys2 = ['two','one','one','three','one','two','three']
# df = pd.DataFrame({'key1':keys1,'key2':keys2, 'year':years, 'value1':x, 'value2':y})
# print (df)
# print('')
# group1 = df.groupby('key1')['value1']
# group2 = df.groupby('key1')[['value1', 'value2']]
# print(group1.mean())
# print(group2.mean())


# import pandas as pd
# df1 = pd.DataFrame ({'key':list('bbaca'), 'data1':range(5)})
# df2 = pd.DataFrame ({'key':['a','b','d'], 'data2':range(3)})
# print(df1)
# print(df2)

# ret = pd.merge(df1, df2, how = 'outer', on = 'key')
# print(ret)

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# df = pd.DataFrame(np.random.rand(10, 4), columns=["a", "b", "c", "d"])
# df.plot.bar()
# plt.show()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# df = pd.DataFrame(np.random.rand(10, 4), columns=["a", "b", "c", "d"])
# df.iloc[5].plot.bar()
# plt.show()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# df = pd.DataFrame(np.random.rand(10, 4), columns=["a", "b", "c", "d"])
# df.plot.bar(stacked=True)
# plt.show()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# df = pd.DataFrame(np.random.rand(10, 4))
# df.plot.hist()
# plt.show()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# df = pd.DataFrame(3 * np.random.rand(4, 2),
#                   index=['a', 'b', 'c', 'd'], columns=['x', 'y'])
# df.plot.pie(subplots=True)
# plt.show()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# df = pd.DataFrame(np.random.rand(10, 4))
# df.plot.box()
# plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame(np.random.rand(10, 4), columns=["a", "b", "c", "d"])
df.plot.scatter(x="a", y="b")
plt.show()
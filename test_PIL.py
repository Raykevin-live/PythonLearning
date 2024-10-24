import PIL as pil 
import numpy as np 
import matplotlib.pyplot as plt 

# img = np.random.random([300, 400])
# plt.imshow(img)
# plt.show()

# from PIL import Image
# image = Image.open('cat.png')
# image.show()
# im = np.array(image)
# print('shape: ', im.shape)

# import numpy as np
# from PIL import Image
# image = Image.open('cat.png')
# image = np.array (image)
# print (image)
# print ('shape:',image.shape)
# print ('dtype:',image.dtype)

# 用 PIL 获取图片 RGB 数值
# from PIL import Image
# im = Image.open('cat.png')
# width = im.size[0]
# height = im.size[1]
# array = []
# for x in range(5):
#     for y in range(5):
#         r, g, b = im.getpixel((x,y))
#         rgb = (r, g, b)
#         array.append(rgb)
# print(array)

# from PIL import Image
# img = Image.open('cat.png')
# img = img.resize((104, 169))
# img.show()

# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# img = Image.open('cat.png')
# img = np.array(img)
# img = img[0:120,100:250]
# plt.imshow(img)
# plt.show()


# import numpy as np
# from PIL import Image
# image = Image.open('cat.png')
# image = np.array(image)
# image1 = image*0.5
# image1 = np.clip(image1, a_min = 20, a_max = 200)
# plt.imshow(image1.astype('uint8'))
# plt.show()

# from PIL import Image
# image = Image.open('cat.png')
# image = np.array (image)
# H,W = image.shape[0],image.shape[1]
# H1 = H // 2 
# H2 = H
# image = image[H1:H2, ::]
# plt.imshow(image)
# plt.show()

import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
im = Image.open('qe.png')
conF = im.filter(ImageFilter.CONTOUR)
conF.show()
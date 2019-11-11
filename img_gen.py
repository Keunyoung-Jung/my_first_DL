import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

generator = ImageDataGenerator(
    rotation_range = 40, #각도
    width_shift_range = 0.1,  #확대 축소
    height_shift_range = 0.1)  #확대 축소

gen2 = []
gen3 = []
fld = 'dataset2/9240202000/' #이미지가 있는 폴더
make_num = 4  #생산할 이미지 갯수
c1 = os.listdir(fld)

for i in range(len(c1)):
    aa = plt.imread(fld+c1[i])
    aa = aa.reshape((1,)+aa.shape)
    a1 = generator.flow(aa)
    for i in range(make_num):
        a3 = a1.next()[0]
        a3 = np.int32(a3)
        gen2.append(a3)
        gen3.append(i)
#gen2 에 이미지가 있습니다 순서대로..
#gen3 는 라벨 이름입니다. (필요시사용)
        
plt.imshow(gen2[5])
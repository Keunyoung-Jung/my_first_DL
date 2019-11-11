import numpy
import cv2
import matplotlib.pyplot as plt
import os

a = os.listdir('natural_images/')

img = []
label = []

for i in range(len(a)):
    file_list = os.listdir('./natural_images/' + a[i])
    for j in range(len(file_list)):
        try:
            c = cv2.imread('./natural_images/'+a[i]+'/'+file_list[j])
            c = cv2.resize(c, dsize = (30, 30))
            img.append(c)
            label.append(file_list[j])
            
            print(c.shape)
            #c = cv2.cvtColor(c,cv2.COLOR_BGR2RGB)
            cv2.imwrite('./natural_images/'+a[i]+'/'+file_list[j],c)
        except:
            pass
import tensorflow as tf
import numpy as np

def standRegres(xArr , yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0.0 :
        print('This matrix is singular , cannot do inverse')
        return
    ws = xTx.I*(xMat.T*yMat)
    print('xMat.T\n',xMat.T)
    print('xTx\n',xTx)
    print('xTx.I\n',xTx.I)
    print('yMat\n',yMat)
    print('xMat.T*yMat\n',xMat.T*yMat)
    return ws

xArr = [[1,1],
        [1,2],
        [1,3],
        [1,4]]
yArr = [3,5,7,9]

ws = standRegres(xArr, yArr)
print('result\n',ws) #바이어스와 기울기가 나온다
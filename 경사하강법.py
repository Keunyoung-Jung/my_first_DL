import random


def gradient_Regress(xArr,yArr,learning_rate=0.01):
    a = random.gauss(0,1)
    b = random.gauss(0,1)
    
    while True:
        for i in range(len(yArr)):
            e = xArr[i][0] # i 번째 데이터의 y 절편
            x = xArr[i][1] # i 번째 데이터의 x 좌표
            y = yArr[i]    # i 번째 데이터의 y 좌표
            y_hat = a*x + b*e   # y절편(1)*b + x좌표*a (target function)
    
            loss = (y-y_hat)*(y-y_hat) # 제곱을 하는 이유는 양수로 만들어야함. (경사하강법을 적용하기 위해서는 양음을 파악해야한다.)
            
            gradient_a = 2*(y-y_hat)*(-x) # a에 대한 편미분
            a -= gradient_a*learning_rate # a에 대해 경사하강법 적용
            
            gradient_b = 2*(y-y_hat)*(-e) # b에 대한 편미분
            b-=gradient_b*learning_rate   # b에 대해 경사하강법 적용
            
        if loss < 0.0001:                 # 에러율이 특정 수치 미만일 경우
            return a, b
            break
        
xArr=[[1,0],[1,1],[1,2],[1,3],[1,4]] 
yArr=[7,9,11,13,15]

gradient_Regress(xArr, yArr)
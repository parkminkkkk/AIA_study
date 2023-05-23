
x = 10
y = 10
w = 11
lr = 0.01 
epochs = 1001

for i in range(epochs):
    hypothesis = x * w 
    loss = (hypothesis -y) **2      #mse        

    print("Loss:", round(loss, 4), '\tPredict:', round(hypothesis, 4))

    #lr의 방향성
    up_predict = x * (w + lr)
    up_loss = (y - up_predict)**2

    down_predict = x * (w - lr)
    down_loss = (y - down_predict)**2
    
    if(up_loss >= down_loss):
        w = w - lr
    else: 
        w = w + lr


# [실습] 얼리스탑핑 적용
#1. 최솟값을 넣을 변수 하나, 카운트 할 변수 하나  (총 변수 두개) 준비!!
#2. 다음 epoch의 값과 최솟값을 비교하여 최솟값이 갱신 된다면 그 변수에 최솟값을 넣어주고, 카운트변수 초기화
#3. 갱신이 안되면 카운트변수 ++1
#   카운트 변수가 내가 원하는 얼리스타핑 개수에 도달하면 for문을 stop 


x = 10
y = 10
w = 11
lr = 0.01
epochs = 100001

best_loss = float('inf')
best_epoch = 0

for i in range(epochs):
    hypothesis = x * w
    loss = (hypothesis - y) ** 2  # MSE

    print("Loss:", round(loss, 4), '\tPredict:', round(hypothesis, 4))

    # 현재 loss, 최고loss비교 
    if loss < best_loss:
        best_loss = loss
        best_epoch = i

    # lr방향성 
    up_predict = x * (w + lr)
    up_loss = (y - up_predict) ** 2

    down_predict = x * (w - lr)
    down_loss = (y - down_predict) ** 2

    if up_loss >= down_loss:
        w = w - lr
    else:
        w = w + lr

    # EarlyStopping
    if i - best_epoch > 10:  # paitence 10
        print("Early stopping at epoch", i)
        break


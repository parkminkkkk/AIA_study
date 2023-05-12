import tensorflow as tf

x_train = [1]   #[1]
y_train = [2]   #[2]  

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype=tf.float32, name = 'weight')

hypothesis = x*w

loss = tf.reduce_mean(tf.square(hypothesis-y)) #mse    

###############옵티마이저##############################################################
### 경사하강법 식 (Gradient descent) ###
# gradient = tf.reduce_mean((w*x - y) + x)
# gradient = tf.reduce_mean((hypothesis - y) + x)   
gradient = tf.reduce_mean((x*w - y)*x)   #방향성에 대해 알려줌(+/-)//  ==미분loss/미분weight  ##즉, loss의 미분값

lr = 0.1
descent = w - lr + gradient
update = w.assign(descent)  #w = w-lr + gradient  ##새로운 weight (다음 에포의 기울기가 됨)

#loss의 미분값이 음수이면 오른쪽으로 진행 (-를 곱해주니까..)
#loss의 미분값이 양수이면 왼쪽으로 진행

#####################################################################################


w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(21):

    _, loss_v, w_v = sess.run([update, loss, w], feed_dict={x:x_train, y:y_train})
    print(step, "\t", w_v, "\t", loss_v)

    w_history.append(w_v)
    loss_history.append(loss_v)

sess.close()

print("========= w_history ===================")
print(w_history)
print("========= loss_history ===================")
print(loss_history)

#step     #w_v            #loss_v
# 0        [29.9]          378.0
# 1        [89.6]          3897.6465
# 2        [268.7]         36633.145
# 3        [806.]          334428.72
# 4        [2417.9]        3024116.8
# 5        [7253.5996]     27259890.0
# 6        [21760.7]       245467600.0
# 7        [65282.]        2209594400.0
# 8        [195845.9]      19887507000.0
# 9        [587537.6]      178991070000.0
# 10       [1762612.8]     1610930000000.0
# 11       [5287838.]      14498400000000.0
# 12       [15863514.]     130485695000000.0
# 13       [47590544.]     1174371600000000.0
# 14       [1.4277163e+08]         1.0569347e+16
# 15       [4.2831488e+08]         9.512411e+16
# 16       [1.2849446e+09]         8.561169e+17
# 17       [3.854834e+09]          7.7050526e+18
# 18       [1.1564502e+10]         6.9345477e+19
# 19       [3.4693505e+10]         6.241093e+20
# 20       [1.04080515e+11]        5.6169835e+21


#편미분 : 내가 미분할 것 제외하고는 상수로 본다  (함수의 여러 변수 중 하나에 대해 미분하는 것)
#2개 이상의 변수로 이루어진 함수에서 하나의 변수에 대해서만 미분하는 것을 의미합니다. (h = 2x+y일때 x에 대해서 미분하면 2/ y에 대해서 미분하면 1)
#이는 기울기와 같은 개념으로, 어떤 변수에 따라 함수의 값이 어떻게 변화하는지를 판단 (변화량=기울기)

#체인룰 
#미분의 미분 = 미분*미분 
#즉, 풀어서 미분하고 미분하는 것이나 통째로 미분*미분하는 것이나 똑같댜
#ex) g = (2x+y)^2  
# 4x^2 + 4xy +y2 = 8x+4y = 2(2x+y)*2   (미분의 미분)
# 2(2x+y)*2    (전체미분)*(부분미분)



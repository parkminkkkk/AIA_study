import tensorflow as tf
tf.compat.v1.set_random_seed(123)

변수 = tf.compat.v1.Variable(tf.random_normal([2], name = 'weigth'))
print(변수)   #<tf.Variable 'Variable:0' shape=(2,) dtype=float32_ref>   
#tf.Variable([2])변수에 들어가는 것은 값/ random_mormal([1])에 들어가는 것은 shape!!

#초기화 첫번째 방법.
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(변수)
print('aaa:', aaa)      #aaa: [-1.5080816   0.26086742]
sess.close()

#초기화 두번째 방법.
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = 변수.eval(session=sess)   #텐서플로 데이터형인 '변수'를 파이썬에서 볼 수 있는 형태로 바꿔 줌 
print("bbb:", bbb)     #bbb: [-1.5080816   0.26086742]
sess.close()

#초기화 세번재 방법. 
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval()
print('ccc:', ccc)    #ccc: [-1.5080816   0.26086742]
sess.close()


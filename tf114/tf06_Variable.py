###텐서1 변수 차이점!!###
#변수를 항상 초기화 해줘야한다!!

import tensorflow as tf
sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype = tf.float32)   #Variable : 변수 // 텐서 1에서는 변수 사용하기 전에 초기화 해줘야함 
y = tf.Variable([3], dtype = tf.float32)   
init = tf.compat.v1.global_variables_initializer()    #전역변수에 대해 초기화시킴// 텐서1 기본 문법이라 생각하면 됨 
sess.run(init)

print(sess.run(x+y))
#[5.]




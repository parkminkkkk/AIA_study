#eager : 즉시 

import tensorflow as tf
print(tf.__version__)

#즉시 실행 모드 
print(tf.executing_eagerly())  #tf114cpu(그래프연산방식) : False //=>>  ##tf273cpu(즉시실행방식) : True

tf.compat.v1.disable_eager_execution()    #즉시실행모드 False(끄기) / 텐서2.0을 텐서1.0방식으로 사용(True->False)

print(tf.executing_eagerly())   #False

tf.compat.v1.enable_eager_execution()     #즉시실행모드 True(켜기) / 텐서1.0을 2.0방식으로 

print(tf.executing_eagerly())   #True

aaa = tf.constant("hello world")

sess = tf.compat.v1.Session() 
print(sess.run(aaa))           #tf2에서는 session없어짐, 그냥 print(aaa)출력 : 즉시실행모드 = 텐서플로2

print(aaa)                     #텐서플로2 즉시실행 
 



# print(tf.executing_eagerly())
# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()      
# sess = tf.compat.v1.Session() 
# print(sess.run(aaa))
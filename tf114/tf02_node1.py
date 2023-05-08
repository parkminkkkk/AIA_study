import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #WARNING 수준 이상의 로그만 출력되며, INFO 수준의 로그는 출력되지 않습니다.

###노드연산방식### 
import tensorflow as tf
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)

# node3 = node1 + node2
node3 = tf.add(node1, node2)   #둘다 가능 

print(node1)   #Tensor("Const:0", shape=(), dtype=float32)
print(node2)   #Tensor("Const_1:0", shape=(), dtype=float32)
print(node3)   #Tensor("add:0", shape=(), dtype=float32)   #(더하기연산, shape, dtype)
#그래프의 모양이 나옴 

#넣어야 할 부분을 모두 'node'로 만들어줌 이후, 출력하고 싶은 부분 출력(sess.run)
sess = tf.compat.v1.Session() 
print(sess.run(node3))   #7.0
print(sess.run(node1))  #3.0



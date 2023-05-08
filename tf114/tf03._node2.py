import tensorflow as tf
node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

#[실습]
#더하기 node3
#빼기 node4
#곱하기 node5
#나누기 node6

node3 = tf.add(node1, node2)
node4 = tf.subtract(node1, node2)
node5 = tf.multiply(node1, node2)
node6 = tf.divide(node1, node2)
# node3 = node1 + node2
# node4 = node1 - node2
# node5 = node1 * node2
# node6 = node1 / node2 

print(node1)   #Tensor("Const:0", shape=(), dtype=float32)
print(node2)   #Tensor("Const_1:0", shape=(), dtype=float32)
print(node3)   #Tensor("add:0", shape=(), dtype=float32)   #(더하기연산, shape, dtype)
print(node4)   #Tensor("sub:0", shape=(), dtype=float32)
print(node5)   #Tensor("mul:0", shape=(), dtype=float32)
print(node6)   #Tensor("truediv:0", shape=(), dtype=float32)


sess = tf.compat.v1.Session() 
print(sess.run(node3))   #5.0
print(sess.run(node4))   #-1.0
print(sess.run(node5))   #6.0
print(sess.run(node6))   #0.6666667

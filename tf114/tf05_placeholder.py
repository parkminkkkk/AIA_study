import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())    #true

#즉시 실행모드
tf.compat.v1.disable_eager_execution()  #꺼 (False) 
print(tf.executing_eagerly())    # False = 텐서1상태

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node4 = tf.add(node1, node2)

sess = tf.compat.v1.Session()

a = tf.compat.v1.placeholder(tf.float32)   #placeholder : 빈 공간에 입력값을 받을 준비하는 곳
b = tf.compat.v1.placeholder(tf.float32) 

add_node = a + b 

print(sess.run(add_node, feed_dict={a:3, b:4.5}))  # placeholder안에 넣을 값을 feed_dict에 키,밸류 형태로 넣어준다// #딕셔너리-키,밸류
#7.5

print(sess.run(add_node, feed_dict={a:[1,3], b:[2,4]}))
#[3. 7.]


#add_node값에 추가
add_and_trple = add_node*3
print(add_and_trple)  #Tensor("mul:0", dtype=float32)
print(sess.run(add_and_trple, feed_dict={a:7, b:3}))
#30.0






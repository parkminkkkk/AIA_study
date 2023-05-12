# tf1 : sess.run부분을 추가해줘야함 (세션 생성한 후 run에서 aaa를 집어넣어줘야 hello world 출력됨)
# sess.run : 그래프 연산 방식(tf1)
## 'tensorflow mechanics' 이미지 참고
# 텐서1의 기본모드 = 그래프 방식**// 텐서2의 기본모드 = 즉시 실행
# 행렬 연산 

import tensorflow as tf
print(tf.__version__)

print("hello world")

aaa = tf.constant("hello world")  #constant:상수(바뀌지않는 숫자)
print(aaa)  #Tensor("Const:0", shape=(), dtype=string)

# sess = tf.Session()   # tf1 : sess단계 거쳐야함 
'''
WARNING:tensorflow:From c:\AIA_study\AIA_study\tf114\tf01_hello.py:13: 
The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.
'''
sess = tf.compat.v1.Session() #버전차이로 session위치 바뀜 
print(sess.run(aaa))  # b'hello world'



 
#####현재 버전이 1.0 이면 그냥 출력/ 현재 버전이 2.0이면 즉시 실행 모드를 끄고 출력#####
#####[if문 사용해서 1번 소스 변경_버전명시]

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
# Tensorflow 버전 확인
print(tf.__version__)

# 즉시 실행 모드가 꺼져 있는 경우에만 disable_eager_execution 함수 호출
if not tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()

# 즉시 실행 모드 확인
print(tf.executing_eagerly())

# 텐서 생성
aaa = tf.constant("hello world")

# Session 실행
if not tf.executing_eagerly():
    sess = tf.compat.v1.Session()
    print(sess.run(aaa))
else:
    print(aaa)


# 2.7.3
# True
# tf.Tensor(b'hello world', shape=(), dtype=string)

'''
#버전확인
import tensorflow as tf
print(tf.__version__)

#버전에 따라 즉시 실행모드 조정
if tf.__version__ == "1.14.0": 
    tf.compat.v1.disable_eager_execution()
elif tf.__version__=="2.7.3":
    tf.compat.v1.enable_eager_execution()

#즉시 실행모드 확인
print(tf.executing_eagerly())

aaa = tf.constant("hello world")

#Session 실행 
if tf.__version__ == "1.14.0": 
    sess = tf.compat.v1.Session()
    print(sess.run(aaa))
elif tf.__version__=="2.7.3":
    print(aaa)
'''
 




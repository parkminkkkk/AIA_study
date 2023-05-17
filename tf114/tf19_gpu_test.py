import tensorflow as tf
tf.compat.v1.set_random_seed(337)  #1.X모드에서 이거 사용
tf.random.set_seed(337)   # 2.X모드에서 이거 사용 

# tf.compat.v1.disable_eager_execution() #즉시모드 안해 1.0 False
# tf.compat.v1.enable_eager_execution() #즉시모드 해 2.0    True

print("텐서플로 버전 :", tf.__version__)
print("즉시실행 모드 :", tf.executing_eagerly())



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:                        #에러 뜰 수도 있는 상황이 있을까봐 애매할떄 try
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print(gpus[0])
    except RuntimeError as e :  #예외가 발생하면 실시간에러를 보여줘 
        print(e)
else : 
    print("gpu 없다!!!")

'''
텐서플로 버전 : 2.7.4
즉시실행 모드 : True 
PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
-------------------------------------------------------------------
텐서플로 버전 : 1.14.0
즉시실행 모드 : False
gpu 없다!!!
'''
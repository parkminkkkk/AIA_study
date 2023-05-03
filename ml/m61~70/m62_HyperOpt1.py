# hyperopt : 최솟값 찾는 알고리즘!!! // BayesianOptimizatio : 최댓값 찾기

# pip install hyperopt
# print(hyperopt.__version__) #0.2.7 // 버전별로 import되는게 조금씩 다름
import numpy as np
import hyperopt
from hyperopt import hp

#1. 파라미터 범위(모아놓은것)
search_space = {
    'x1' : hp.quniform('x1', -10, 10, 1),  #-10부터 10까지 1단위로 찾아라  
    'x2' : hp.quniform('x2', -15, 15, 1)
           #hp.quniform(label, low, high, q) #hp.quniform : q분포로 search를 하겠다 (정수형태일때)
}
# print(search_space)

# hp.uniform(label, low, high) : 최소부터 최대까지 정규분포 간격
# hp.quniform(label, low, high, q) : 최소부터 최대까지 q간격
# hp.randint(label, upper) : 0부터 최댓값upper(지정)까지 random한 정수값
# hp.loguniform(label, low, high) : exp(uniform(low, high))값 반환  /이것 또한 정규분포 /  log변환 한것을 다시 지수로 변환한다(exp)
# *x : 독립변수 (x의 값이 너무 크거나 치우쳐져있는 경우에도 log변환) / y : 종속변수( 주로 y값 log변환) 

#2. 목적함수 정의
def objective_func(search_space):
    x1 = search_space['x1']          #search_space 분리해줘야함 
    x2 = search_space['x2']
    return_value = x1**2 -20*x2
    
    return return_value             #권장리턴방식 : return {'loss':return_value, 'status': STATUS_OK}

#3. 최솟값 찾기
from hyperopt import fmin, tpe, Trials, STATUS_OK
#fmin : 최솟값 찾는 함수/ #tpe : 알고리즘 정의 /#Trials : 결과값 저장 (history와 비슷)

trial_val = Trials()

best = fmin(
    fn= objective_func,                         #함수
    space= search_space,                        #파라미터
    algo=tpe.suggest,                           #알고리즘 정의(디폴트) // 베이지안 최적화와의 차이점..
    max_evals=20,                               #베이지안 최적화의 n_iter와 동일(훈련 10번)
    trials=trial_val,                           #결과값 저장
    rstate = np.random.default_rng(seed=10)    #random_state와 동일
)

print("best:", best)
# best: {'x1': 0.0, 'x2': 15.0}
print(trial_val.results)
# [{'loss': -216.0, 'status': 'ok'},..., {'loss': 0.0, 'status': 'ok'}]
print(trial_val.vals)
# {'x1': [-2.0, -5.0, 7.0, 10.0, 10.0, 5.0, 7.0, -2.0, -7.0, 7.0, 4.0, -7.0, -8.0, 9.0, -7.0, 0.0, -0.0, 4.0, 3.0, -0.0], 
# 'x2': [11.0, 10.0, -4.0, -5.0, -7.0, 4.0, -8.0, 9.0, 3.0, 5.0, -6.0, 5.0, -5.0, -12.0, 0.0, 15.0, -8.0, 7.0, 1.0, 0.0]}


#####[실습]trial_val.vals를 pd.DataFrame에 넣기#############
# df_trial = pd.DataFrame(trial_val.vals, columns = ['x1', 'x2'])
# print(df_trial)

#####[yys]trial_val.vals를 pd.DataFrame에 넣기#############
import pandas as pd
# 동일 코드
# for aaa in trial_val.results:
#     losses.append(aaa['loss'])

results = [aaa['loss'] for aaa in trial_val.results]   #trial_val.results의 값을 aaa반복해라 / aaa의 ['loss']만 반복해라 

df = pd.DataFrame({'x1': trial_val.vals['x1'],
                   'x2': trial_val.vals['x2'],
                   'results': results})
print(df)
'''
      x1    x2  results
0   -2.0  11.0   -216.0
1   -5.0  10.0   -175.0
2    7.0  -4.0    129.0
3   10.0  -5.0    200.0
...
13   9.0 -12.0    321.0
14  -7.0   0.0     49.0
15   0.0  15.0   -300.0  ***best***
16  -0.0  -8.0    160.0
17   4.0   7.0   -124.0
18   3.0   1.0    -11.0
19  -0.0   0.0      0.0
'''

#====================================================================#
###best###
#seed = 37
#max_evals=10 : best: {'x1': 5.0, 'x2': 8.0}
#max_evals=20 : best: {'x1': -1.0, 'x2': 11.0}
#max_evals=40 : best: {'x1': -1.0, 'x2': 15.0}
#max_evals=148 : best: {'x1': 0.0, 'x2': 15.0} **정답**

#seed=10
#max_evals=20 : best: {'x1': 0.0, 'x2': 15.0} **정답**
#====================================================================#

###trial_val.results###
'''
[{'loss': -216.0, 'status': 'ok'}, {'loss': -175.0, 'status': 'ok'}, 
{'loss': 129.0, 'status': 'ok'}, {'loss': 200.0, 'status': 'ok'}, 
{'loss': 240.0, 'status': 'ok'}, {'loss': -55.0, 'status': 'ok'}, 
{'loss': 209.0, 'status': 'ok'}, {'loss': -176.0, 'status': 'ok'}, 
{'loss': -11.0, 'status': 'ok'}, {'loss': -51.0, 'status': 'ok'}, 
{'loss': 136.0, 'status': 'ok'}, {'loss': -51.0, 'status': 'ok'}, 
{'loss': 164.0, 'status': 'ok'}, {'loss': 321.0, 'status': 'ok'}, 
{'loss': 49.0, 'status': 'ok'}, {'loss': -300.0, 'status': 'ok'}, 
{'loss': 160.0, 'status': 'ok'}, {'loss': -124.0, 'status': 'ok'},
{'loss': -11.0, 'status': 'ok'}, {'loss': 0.0, 'status': 'ok'}]
'''

###trial_val.vals###
'''
{'x1': [-2.0, -5.0, 7.0, 10.0, 10.0, 5.0, 7.0, -2.0, -7.0, 7.0, 4.0, -7.0, -8.0, 9.0, -7.0, 0.0, -0.0, 4.0, 3.0, -0.0], 
'x2': [11.0, 10.0, -4.0, -5.0, -7.0, 4.0, -8.0, 9.0, 3.0, 5.0, -6.0, 5.0, -5.0, -12.0, 0.0, 15.0, -8.0, 7.0, 1.0, 0.0]}
'''

###df_trial###
'''
      x1    x2
0   -2.0  11.0
1   -5.0  10.0
2    7.0  -4.0
3   10.0  -5.0
4   10.0  -7.0
5    5.0   4.0
6    7.0  -8.0
7   -2.0   9.0
8   -7.0   3.0
9    7.0   5.0
10   4.0  -6.0
11  -7.0   5.0
12  -8.0  -5.0
13   9.0 -12.0
14  -7.0   0.0
15   0.0  15.0  ***best***
16  -0.0  -8.0
17   4.0   7.0
18   3.0   1.0
19  -0.0   0.0
'''
# hyperopt : 최솟값 찾는 알고리즘!!! // BayesianOptimizatio : 최댓값 찾기

# pip install hyperopt
# print(hyperopt.__version__) #0.2.7 // 버전별로 import되는게 조금씩 다름
import numpy as np
import pandas as pd
import hyperopt
from hyperopt import hp

#1. 파라미터 범위(모아놓은것)
search_space = {
    'x1' : hp.quniform('x1', -10, 10, 1),  #-10부터 10까지 1단위로 찾아라  
    'x2' : hp.quniform('x2', -15, 15, 1)
           #hp.quniform(label, low, high, q) #hp.quniform : 정규분포에서 q분포로 search를 하겠다
}
# print(search_space)


#2. 목적함수 정의
def objective_func(search_space):
    x1 = search_space['x1']          #search_space 분리해줘야함 
    x2 = search_space['x2']
    return_value = x1**2 -20*x2
    
    return return_value

#3. 최솟값 찾기
from hyperopt import fmin, tpe, Trials  
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
print(trial_val.results)
print(trial_val.vals)

#[실습]trial_val.vals를 pd.DataFrame에 넣기
df_trial = pd.DataFrame(trial_val.vals, columns = ['x1', 'x2'])
print(df_trial)

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
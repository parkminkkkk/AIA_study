# 결측치 처리 
'''
1. 삭제(행, 열) 
2. 특정 값 대체
    1) 평균값 mean
    2) 중위값(중앙값) median
    3) 0   : fillna
    4) 앞값 : ffill  (주로, 시계열데이터)
    5) 뒷값 : bfill  
    6) 특정값 : ...
    7) 기타 등등...
3. 보간(interpolation)
 -선형 모델을 만들어서 선을 하나 긋겠다 (모델 만들어서 predict하는 것과 비슷)
 -선을 하나 그었을때 결측치가 선의 어디 지점에 위치하는지 
4. 모델 : predict 
 - 결측치 데이터만 따로 제외시킴 => 이후 x,y를 합쳐서 X, 결측치가 존재하는 열을 Y로 잡음 => 이후 predict으로 결측치 찾기 
 - 과적합 조심
5. 트리/부스팅 계열 
 -통상 결측치, 이상치에 대해 자유롭다. (결측치 존재햐도 모델 훈련,평가 가능) 
'''

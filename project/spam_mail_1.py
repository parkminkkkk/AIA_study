import numpy as np
import pandas as pd

path = './_data/project/'
save_path = '/_save/project/'

dt_eng = pd.read_csv(path + 'spam_ham_dataset.csv')
#dt_kor = pd.read_csv(path)

#data concat(pd.concat)

print(dt_eng)
print(dt_eng.shape) #(5171, 4)
email = dt_eng["text"]

print("이메일 최대길이", max(len(i) for i in email)) #이메일 최대길이 32258
print("이메일 평균길이", )
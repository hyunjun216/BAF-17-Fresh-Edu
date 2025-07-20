import pandas as pd
import seaborn as sns
tt =sns.load_dataset('titanic')

# 데이터 타입 확인
type(tt) # 데이터 프레임
type(tt['pclass']) # 시리즈
tt['pclass']

sum(tt['pclass']) # 총합
min(tt['pclass']) #  최솟값
max(tt['pclass']) # 최댓값
import numpy as np
np.mean(tt['pclass']) # 산술평균


tt.head(8)
tt.tail(8)

tt.shape # 전체 행, 열 추출  
tt.info() # 데이터 요약
tt.columns #  컬럼만 출력
tt.dtypes # 데이터 타입 확인
tt.describe(include='all') # 통계 요약정보 확인



t_num = tt.select_dtypes(include='number') # 수치형 데이터만 추출 
np.mean(tt['pclass']) # 산술평균
np.median(tt['pclass']) # 중앙값

tt['pclass'].mode() # 최빈값
 
np.std(tt['pclass']) # 표준편차
np.var(tt['pclass']) # 분산
tt['pclass'].quantile([0.25, 0.5, 0.75]) # 1,2,3 분위수
tt[['pclass', 'survived']].corr().iloc[0,1]  # 상관계수
tt[['pclass', 'survived']].cov().iloc[0,1] # 공분산

from scipy import stats
stats.skew(tt['pclass']) # 왜도
stats.kurtosis(tt['pclass']) # 첨도
stats.sem(tt['pclass']) # 표준오차

import matplotlib.pyplot as plt
null_count =tt.isnull().sum() # 결측치 값 합쳐서 확인하기
null_count
null_count.plot.bar(rot=60) # 결측치 시각화
plt.show()

# 결측치 확인
tt_null_count = null_count.reset_index() 
tt_null_count.columns = ["컬럼명", "결측치수"]
tt_null_count_top = tt_null_count.sort_values(by="결측치수", ascending=False).head(5) # ascending=False 결측치수 많은 수로 정
tt_null_count_top

# 'deck' 결측치 너무 많아서 삭제
tt.drop(columns=['deck'], inplace=True)

# 'age' 결측치를 중앙값으로 대체
tt['age'].fillna(tt['age'].median(), inplace=True)

# 'embarked'와 'embark_town' 결측치를 최빈값으로 대체
tt['embarked'].fillna(tt['embarked'].mode()[0], inplace=True)
tt['embark_town'].fillna(tt['embark_town'].mode()[0], inplace=True)

null_count =tt.isnull().sum() # 결측치 값 합쳐서 확인하기



 


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()  # 객체 생성
tt['Sex_encoded'] = le.fit_transform(tt['sex']) 
# 레이블 인코딩 여자 : 0, 남자 : 1
tt['Sex_encoded']

# 필요없는 데이터 삭제
tt.drop(columns=['sex'], inplace=True)
tt


# 범주형 데이터 열 출력
non_numeric_cols = tt.select_dtypes(exclude=['number']).columns
print(non_numeric_cols)


 
# 불리언 컬럼 0/1로 변환
for col in ['adult_male', 'alone']:
    tt[col] = tt[col].astype(int)

#  alive 레이블 인코딩 no -> 0, yes -> 1
le = LabelEncoder()
tt['alive_encoded'] = le.fit_transform(tt['alive'])
tt.drop(columns=['alive'], inplace=True)

#  embarked, class, who, embark_town 원-핫 인코딩 -> 데이터 간 순서가 있는 경우
tt = pd.get_dummies(tt, columns=['embarked', 'class', 'who', 'embark_town'], drop_first=True)
tt

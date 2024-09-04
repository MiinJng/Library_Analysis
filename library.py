import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# 1. 데이터 준비
data = pd.read_csv('/Users/minjeong/Downloads/lib_data.csv')

# 2. 범주형 변수 더미화
X = pd.get_dummies(data[['program_target', 'program_day', 'program_type']], drop_first=False)

# bool 타입을 int로 변환
X = X.astype(int)

# 종속 변수
y = data['loan_change']

# 3. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. sklearn을 사용한 회귀 모델
model = LinearRegression()
model.fit(X_train, y_train)

# 회귀 계수 출력 (sklearn)
print("Sklearn Coefficients:", model.coef_)
print("Sklearn Intercept:", model.intercept_)

# 5. statsmodels를 사용한 회귀 모델
X_train_with_const = sm.add_constant(X_train)  # 상수 추가
model_sm = sm.OLS(y_train, X_train_with_const).fit()

# 회귀 계수와 p-value 출력 (statsmodels)
model_sm.summary()

coefficients = model_sm.params
p_values = model_sm.pvalues
print("\nCoefficients:\n", coefficients)
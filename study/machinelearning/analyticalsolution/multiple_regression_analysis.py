from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer

"""
https://tutorials.chainer.org/ja/09_Introduction_to_Scikit-learn.html
"""

dataset = load_boston()
x = dataset.data
t = dataset.target

print(x.shape)
print(x[0])
print(t.shape)
print(t[0])

x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=0)

reg_model = LinearRegression()

# learning
reg_model.fit(x_train, t_train)

# weight parameter
print(f'weight: {reg_model.coef_}')

# bias parameter
print(f'bias: {reg_model.intercept_}')

print(f'coefficient of determination by training data: {reg_model.score(x_train, t_train)}')

print(f'predict: {reg_model.predict(x_test[:1])}, actual: {t_test[0]}')

print(f'coefficient of determination by test data: {reg_model.score(x_test, t_test)}')


scaler = StandardScaler()
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

reg_model = LinearRegression()

reg_model.fit(x_train_scaled, t_train)

print(f'{reg_model.score(x_train_scaled, t_train)}')
print(f'{reg_model.score(x_test_scaled, t_test)}')

"""
"""
pipeline = Pipeline([
    ('scaler', PowerTransformer()),
    ('reg', LinearRegression())
])
pipeline.fit(x_train, t_train)
print(f'{pipeline.score(x_train, t_train)}')
print(f'{pipeline.score(x_test, t_test)}')
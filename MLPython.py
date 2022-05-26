import numpy
import matplotlib.pyplot as mp
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

digits=datasets.load_digits()
digits_x=digits.data

digits_x_training=digits_x[:-40]
digits_x_test=digits_x[-40:]

digits_y_traning=digits.target[:-40]
digits_y_test=digits.target[-40:]

model=linear_model.LinearRegression()
model.fit(digits_x_training, digits_y_traning)
digits_y_predicted=model.predict(digits_x_test)

print(f"The Mean squared error is {mean_squared_error(digits_y_test,digits_y_predicted)}")
print(f"Weight is {model.coef_}")
print(f"Intercept is {model.intercept_}")



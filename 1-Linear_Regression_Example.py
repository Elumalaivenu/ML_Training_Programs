
'''simple linear regression formula is y = b0 + b1*x1
y =Dependent Variable (DV)
x =Independent Variable (IV)
b1 =coefficent
b0 =Constant 
'''
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('emp_Salary_Details.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

"Training set → used to train (fit) the model" 
"Testing set → used to evaluate model performance on unseen data" 
"This prevents overfitting and helps estimate how well your model generalizes to new data."

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)'''


# fitting simple Linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


#predicting the test set results
y_pred = regressor.predict(X_test)


# --- Model evaluation ---
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred) #It’s the average magnitude of prediction errors
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Model evaluation metrics:")
print(f"R^2 score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Save trained model to disk using joblib
import joblib
MODEL_PATH = 'models/linear_regressor.joblib'
joblib.dump(regressor, MODEL_PATH)
print(f"Trained model saved to: {MODEL_PATH}")


#visualising the training  set results
plt.scatter(X_train,y_train, color ='red')
plt.plot(X_train, regressor.predict(X_train), color ='blue' )
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Expereience')
plt.ylabel('Salary')
plt.show()

#visualising the test  set results
plt.scatter(X_test,y_test, color ='red')
plt.plot(X_train, regressor.predict(X_train), color ='blue' )
plt.title('Salary vs Experience (test set)')
plt.xlabel('Years of Expereience')
plt.ylabel('Salary')
plt.show()








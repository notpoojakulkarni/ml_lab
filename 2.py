import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Step 1: Import the dataset
# Assuming you have the dataset in a CSV file called 'california_housing.csv'
data = pd.read_csv('housing.csv')

# Step 2: Display the first 5 rows of the dataset
print(data.head())

# Step 3: Check the number of samples of each class (Regression problem, no classes)

# Step 4: Check for null values
print(data.isnull().sum())

# Step 5: Visualize the data using graphs
sns.pairplot(data)
plt.show()

# Step 6: Obtain covariance and correlation values
covariance_matrix = data.cov()
correlation_matrix = data.corr()

print("Covariance Matrix:")
print(covariance_matrix)

print("Correlation Matrix:")
print(correlation_matrix)
data=data.dropna(subset=["total_bedrooms"])
dummies=pd.get_dummies(data.ocean_proximity)
data=pd.concat([data,dummies],axis="columns")
data=data.drop(['ocean_proximity'],axis="columns")

# Step 7: Train and test the regression model
X = data.drop('median_house_value', axis=1)
y = data['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Apply regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Step 9: Predict the accuracy and plot the graph
y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Step 10: Plot the graph
plt.scatter(y_test, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True Values vs. Predictions")
plt.show()


def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        predictions = X.dot(theta)
        error = predictions - y
        gradient = (1 / m) * X.T.dot(error)
        theta -= learning_rate * gradient
    return theta

# Step 3: Train the model on the dataset
learning_rate = 0.01
num_iterations = 1000
theta = np.random.rand(X_train.shape[1])  # Initialize theta with random values

theta = gradient_descent(X_train, y_train, theta, learning_rate, num_iterations)

# Step 4: Predict the prices for new data
def predict(X, theta):
    return X.dot(theta)

# Predicting on the test set
y_pred = predict(X_test, theta)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Import the dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Step 2: Display first 5 rows
print(df.head())

# Step 3: Number of samples in each class (not applicable for regression)

# Step 4: Check for null values
print("Null values in the dataset:")
print(df.isnull().sum())

# Step 5: Visualize the data (Scatter plot of target variable vs. feature)
plt.scatter(df['MedInc'], df['MedHouseVal'])
plt.xlabel('Median Income')
plt.ylabel('Median House Price')
plt.title('Scatter Plot: Median Income vs. Median House Price')
plt.show()

# Step 6: Obtain covariance and correlation values
cov_matrix = df.cov()
corr_matrix = df.corr()

print("Covariance Matrix:")
print(cov_matrix)

print("\nCorrelation Matrix:")
print(corr_matrix)

# Step 7: Split the data into training and testing sets
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train the regression model using Stochastic Gradient Descent
model = SGDRegressor(max_iter=1000, alpha=0.01, random_state=42)
model.fit(X_train, y_train)

# Step 9: Test the model
y_pred = model.predict(X_test)

# Step 10: Predict the accuracy and plot graph
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R-squared (Coefficient of Determination):", r2)

# Plot graph for predicted vs. actual values
plt.scatter(y_test, y_pred)

plt.xlabel('Actual Median House Price')
plt.ylabel('Predicted Median House Price')
plt.title('Actual vs. Predicted Median House Price')
plt.show()
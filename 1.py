import numpy as np
from sklearn.linear_model import LinearRegression

# Dataset
hours_studied = np.array([2, 3, 4, 5, 6]).reshape(-1, 1)
exam_scores = np.array([70, 75, 85, 90, 95]).reshape(-1, 1)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(hours_studied, exam_scores)

# Predict new exam scores for hours_studied = 7
new_hours_studied = np.array([7]).reshape(-1, 1)
predicted_exam_scores = model.predict(new_hours_studied)

print("Predicted exam score for 7 hours studied:", predicted_exam_scores)

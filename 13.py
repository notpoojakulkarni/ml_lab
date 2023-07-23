import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Import the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['Species'])
data['Species'] = data['Species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Step 2: Display first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())

# Step 3: Visualize the data in the form of graphs
plt.figure(figsize=(10, 6))
for species in data['Species'].unique():
    species_data = data[data['Species'] == species]
    plt.scatter(species_data['sepal length (cm)'], species_data['sepal width (cm)'], label=species)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Sepal Length vs. Sepal Width")
plt.legend()
plt.show()

# Step 4: Split the data into training and testing sets
X = data.iloc[:, :-1]
y = data['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train and test the KNN model
k = 3  # Set the number of neighbors for KNN
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)

# Step 6: Apply the KNN classifier
y_pred = knn_classifier.predict(X_test)

# Step 7: Classify the species by providing the test data
test_data = np.array([[5.1, 3.5, 1.4, 0.2],   # Sample test data (features for one instance)
                      [6.3, 2.9, 5.6, 1.8],
                      [7.2, 3.2, 6.0, 1.8]])

predicted_species = knn_classifier.predict(test_data)

print("Predicted Species for Test Data:")
for i in range(len(test_data)):
    print(f"Test instance {i + 1}: {predicted_species[i]}")

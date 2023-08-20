import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neighbors import KNeighborsClassifier
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("IRIS.csv")
df.head(5)
df['Species'].value_counts()
df.isnull().sum()

sns.pairplot(df, hue="Species", size=3)
plt.show()
df.corr()
df.cov()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

from sklearn.model_selection import train_test_split
X = df.drop(columns = ['Species'])
Y = df['Species']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

model =KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, Y_train)


# Make predictions on the test set
Y_pred = model.predict(X_test)

num_predictions = 10
random_indices = np.random.randint(0, len(Y_test), num_predictions)
predicted_species = le.inverse_transform(Y_pred[random_indices])
actual_species = le.inverse_transform(Y_test.values[random_indices])
print("\nRandom Predictions (Predicted Species vs. Actual Species):\n")
for i in range(num_predictions):
    print(f"Prediction {i+1}: {predicted_species[i]} \t Actual: {actual_species[i]}")

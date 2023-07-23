
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Step 1: Create the dataset
data = {
    'VAR1': [1.713, 0.180, 0.353, 0.940, 1.486, 1.266, 1.540, 0.459, 0.773],
    'VAR2': [1.586, 1.786, 1.240, 1.566, 0.759, 1.106, 0.419, 1.799, 0.186],
    'CLASS': [0, 1, 1, 0, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# Step 2: Perform k-means clustering
k = 3
X = df[['VAR1', 'VAR2']]
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Step 3: Predict classification for VAR1=0.906 and VAR2=0.606
new_point = np.array([[0.906, 0.606]])
predicted_cluster = kmeans.predict(new_point)
print("Predicted Cluster:", predicted_cluster[0])

# Step 4: Display the dataset with assigned clusters
print(df)

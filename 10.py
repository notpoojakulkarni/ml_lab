import numpy as np
import pandas as  pd
from sklearn.cluster import KMeans

df={
    'VAR1': [1.713, 0.180, 0.353, 0.940, 1.486, 1.266, 1.540, 0.459, 0.773],
    'VAR2': [1.586, 1.786, 1.240, 1.566, 0.759, 1.106, 0.419, 1.799, 0.186],
    'CLASS': [0, 1, 1, 0, 1, 0, 1, 1, 1]
}

df=pd.DataFrame(df)
X=df[['VAR1','VAR2']]
y=df["CLASS"]

model=KMeans(n_clusters=3)
df["Cluster"]=model.fit_predict(X)

print(df)

new_point=np.array([[0.906,0.606]])
predicted=model.predict(new_point)

print("Predicted cluster: ",predicted[0])
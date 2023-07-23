import pandas as pd
df = pd.read_csv("enjoysport.csv")
print(df)
print(df.shape)
num_attributes = 4
a = []
for i in range(len(df)):
    a.append(df.iloc[i].tolist())
a
hypo = ['?'] * num_attributes
hypo #most specific hypothesis
for i in range(len(a)):
    if a[i][num_attributes] == "Yes":
        # For each attribute constraint ai in h
        for j in range(num_attributes):
            if hypo[j] == '?' or hypo[j] == a[i][j]:
                # Else replace ai in h by the next more general constraint that is satisfied by x
                hypo[j] = a[i][j]
            else:
                # If the constraint ai is not satisfied by x
                hypo[j] = '?'
    print("\n\nFor Training instance No:{0}, the hypothesis is\n".format(i), hypo) 
print("\nThe Maximally Specific Hypothesis for the given Training Examples:\n")
print(hypo)

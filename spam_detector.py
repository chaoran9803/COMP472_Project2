import pandas as pd

df = pd.read_csv('spam.csv', encoding='latin-1')

df = df[['label', 'message']]

print(df.head(5))

print(df.shape)

print(df['label'].value_counts())
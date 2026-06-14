import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('spam.csv', encoding='latin-1')

df = df[['label', 'message']]

print(df.head(5))
print(df.shape)
print(df['label'].value_counts())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
Y = df['label']

print(X.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)

model = MultinomialNB()

model.fit(X_train, Y_train)

print("Finished training the model")
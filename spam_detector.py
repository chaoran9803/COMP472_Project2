import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


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

y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

cm = confusion_matrix(Y_test, y_pred, labels = ['spam', 'ham'])
print("Confusion Matrix:")
print(cm)


test_message = ["Congratulations! You won a free prize. Claim now!"]

test_features = vectorizer.transform(test_message)


prediction = model.predict(test_features)[0]


probabilities = model.predict_proba(test_features)[0]
confidence = max(probabilities) * 100

print(f"Prediction: {prediction.upper()}")
print(f"Confidence: {confidence:.1f}%")


counts = df['label'].value_counts()


plt.bar(counts.index, counts.values, color=['green', 'red'])


plt.title('Spam vs Ham Messages')
plt.xlabel('Label')
plt.ylabel('Number of Messages')


plt.savefig('chart.png')
plt.show()
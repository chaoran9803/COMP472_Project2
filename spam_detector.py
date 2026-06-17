import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer # Convert text to numerical features using TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# load the emails dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['label', 'message']]

# encode
vectorizer = TfidfVectorizer() # encode the emails
X = vectorizer.fit_transform(df['message']) # X is the feature matrix
Y = df['label'] # Y is the target variable
print(X.shape) # it will show (5572, 8674) which means we have 5572 emails and 8674 unique words in the dataset

# split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y) # split the data into training and testing sets, with 20% of the data used for testing. The random_state parameter ensures reproducibility, and stratify ensures that the class distribution is maintained in both sets.
model = MultinomialNB() # create an instance of the Multinomial Naive Bayes classifier
model.fit(X_train, Y_train) # train the model using the training data
print("Model has been trained successfully!!!")

# evaluate the model
y_pred = model.predict(X_test) # predict the labels for the test set
accuracy = accuracy_score(Y_test, y_pred) # calculate the accuracy of the model by comparing the predicted labels with the true labels in the test set
print(f"Accuracy: {accuracy*100:.2f}%") 
cm = confusion_matrix(Y_test, y_pred, labels = ['spam', 'ham'])
print("Confusion Matrix:")
print(cm)


test_message = ["Congratulations! You won a free prize. Claim now!"]
test_features = vectorizer.transform(test_message)

# Make a prediction using the trained model
prediction = model.predict(test_features)[0]
probabilities = model.predict_proba(test_features)[0]
confidence = max(probabilities) * 100
print(f"Prediction: {prediction.upper()}")
print(f"Confidence: {confidence:.1f}%")

# Graphic
counts = df['label'].value_counts()
plt.bar(counts.index, counts.values, color=['green', 'red'])
plt.title('Spam vs Ham Messages')
plt.xlabel('Label')
plt.ylabel('Number of Messages')
plt.savefig('chart.png')
plt.show()
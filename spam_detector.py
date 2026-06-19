import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# 1. Load the emails dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['label', 'message']]

# 2. Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
Y = df['label']
print(f"Dataset loaded. Feature matrix shape: {X.shape}")

# 3. Split the data and Train the Model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

model = MultinomialNB()
model.fit(X_train, Y_train)
print("Training model...")
print("Model has been trained successfully!\n")

# 4. Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy*100:.1f}%") 

cm = confusion_matrix(Y_test, y_pred, labels=['spam', 'ham'])
print("Confusion Matrix:")
print(cm)
print("-" * 30)

# 5. Data Visualization
counts = df['label'].value_counts()
plt.bar(counts.index, counts.values, color=['green', 'red'])
plt.title('Spam vs Ham Messages')
plt.xlabel('Label')
plt.ylabel('Number of Messages')
plt.savefig('chart.png')
print("\nBar chart saved as 'chart.png'.")
print("*** NOTE: Close the chart window to start the interactive prediction loop! ***")
plt.show() 

# 6. Interactive Prediction Loop
print("\nWelcome to Spam Detection AI")
print("Type 'quit' to exit.")

while True:
    # Accept user input
    user_input = input("Enter message:\n")
    
    # Check for exit command
    if user_input.strip().lower() == 'quit':
        print("Goodbye!")
        break
        
    # Convert user input to numerical features
    test_features = vectorizer.transform([user_input])
    
    # Make a prediction using the trained model
    prediction = model.predict(test_features)[0]
    probabilities = model.predict_proba(test_features)[0]
    confidence = max(probabilities) * 100
    
    # Display results
    print(f"Prediction: {prediction.upper()}")
    print(f"Confidence: {confidence:.1f}%\n")
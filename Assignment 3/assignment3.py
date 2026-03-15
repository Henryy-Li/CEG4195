'''
Course:  CEG 4195
Name:    Henry Li

Running instructions:
1. Open a command prompt.
2. Enter "conda init"
3. Enter "conda activate nlp3XX", where XX is the subversion value of Python 3.
   - For Python 3.12, XX = 12 to give us nlp312
4. Enter "python assignment3.py" to run the code. 

Dataset: stanfordnlp/imdb
'''

from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
print("Loading dataset...")
dataset = load_dataset("imdb")
print("Dataset loaded!")

# Parse data
print("Extracting data...")
train = dataset["train"].shuffle(seed=42) # Value of 42 is for reproducable randomness.
test = dataset["test"].shuffle(seed=42)   # Must shuffle as dataset has all negative reviews in the first half and positive reviews in the second half of the dataset.

X_train = train["text"][:2000]            # Limit amount of data so it doesn't run too slow.
y_train = train["label"][:2000]
X_test = test["text"][:2000]
y_test = test["label"][:2000]
print("Data extraction complete!")

# Vectorize the text. Give text values a numerical value.
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000)    # Take the first 5000 most common words
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)
print("Vectorizing of text complete!")

# Inialize and train model
print("Training model...")
model = LogisticRegression(max_iter=1000)          # Let the model learn 1000 before being tested
model.fit(X_train_vector, y_train)
print("Training complete!")

# Make predictions on the test set of data
print("Predicting values...")
y_pred = model.predict(X_test_vector)
print("Predictions complete!")

# Evaluate predictions
print("Evaluating predictions...")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
confusionMatrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy*100:.2f}%")      # Accuracy  = 83.5%
print(f"Precision: {precision*100:.2f}%")    # Precision = 82.09%
print(f"Recall: {recall*100:.2f}%")          # Recall    = 85.70%
print(f"F1-Score: {f1*100:.2f}%")            # F1-Score  = 83.86%

plt.imshow(confusionMatrix)
plt.colorbar()

plt.title("Model Confusion Matrix")
plt.xticks([0,1],["Negative", "Positive"])
plt.yticks([0,1], ["Negative", "Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

for i in range(confusionMatrix.shape[0]):
    for j in range(confusionMatrix.shape[1]):
        plt.text(j,i,confusionMatrix[i,j], ha="center", va="center")
plt.show()
print("Evaluation complete!")





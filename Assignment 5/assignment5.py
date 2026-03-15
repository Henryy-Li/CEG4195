'''
Course:  CEG 4195
Name:    Henry Li

Running instructions:
1. Open a command prompt.
2. Enter "conda init"
3. Enter "conda activate nlp3XX", where XX is the subversion value of Python 3.
   - For Python 3.12, XX = 12 to give us nlp312
4. Enter "python assignment5.py" to run the code. 

Dataset: ylecun/mnist
'''
#=======================================================
from datasets import load_dataset
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
#=======================================================
# ===== Load dataset =====
print("Loading dataset...")
dataset = load_dataset("ylecun/mnist")
print("Dataset loaded!")

# ===== Extract data =====
print("Extracting data...")

train = dataset["train"].shuffle(seed=42)          # Value of 42 is for reproducable randomness.
test = dataset["test"].shuffle(seed=42)   

X_trainDataset = train["image"][:2000]             # Limit amount of data so it doesn't run too slow.
y_train = np.array(train["label"][:2000])
X_testDataset = test["image"][:2000]
y_test = np.array(test["label"][:2000])

print("Data extraction complete!")

# ===== Normazlize pixel values (for images) =====
print("Normalizing pixel values...")

X_train = []                                        # Put all image data into a numpy array
for image in X_trainDataset:
    X_train.append(np.array(image))
X_train = np.stack(X_train)

X_test = []                                         # Put all image data into a numpy array
for image in X_testDataset:
    X_test.append(np.array(image))
X_test = np.stack(X_test)

X_train = X_train.astype('float32') / 255.0         # Normalize
X_test = X_test.astype('float32') / 255.0

print("Normalization complete!")

# ===== Flatten image data into a 1D array =====
print("Flattening data...")
X_train_flat = X_train.reshape(X_train.shape[0], -1)    # Model only takes 1D array inputs.
X_test_flat = X_test.reshape(X_test.shape[0], -1)
print("Flattening complete!")

# ===== Build the deep learning model =====

print("Building model...")
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),  # Each image is 28x28. A 1D array must be 28x28=784.
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
print("Build complete!")

# ===== Compile the model =====

print("Compiling model...")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("Compilation complete!")

# ===== Train the model =====

print("Training model...")
trainValidHistory = model.fit(X_train_flat, y_train, epochs=10, batch_size=32, validation_split=0.2)
print("Training complete!")

# ===== Test the model =====

print("Testing model...")
y_prediction_probabilities = model.predict(X_test_flat)             # Get the probabilitiy of each digit.
y_predictions = np.argmax(y_prediction_probabilities, axis=1)       # Get the digit with the highest probability.
print("Testing complete!")

# ===== Evaluative metrics: Classification metrics =====

print("Evaluating model (Classification metrics)...")

loss, accuracy = model.evaluate(X_test_flat, y_test)
precision = precision_score(y_test, y_predictions, average='weighted')
recall = recall_score(y_test, y_predictions, average='weighted')
f1 = f1_score(y_test, y_predictions, average='weighted')
confusionMatrix = confusion_matrix(y_test, y_predictions)

print(f"Accuracy: {accuracy*100:.2f}%")      # Accuracy  = ~90.95% 
print(f"Loss: {loss:.4f}")                   # Loss      = ~0.3513
print(f"Precision: {precision*100:.2f}%")    # Precision = ~91.55%
print(f"Recall: {recall*100:.2f}%")          # Recall    = ~90.95%
print(f"F1-Score: {f1*100:.2f}%")            # F1-Score  = ~90.98%

plt.figure(figsize=(8,5))                                                              # Loss curves
plt.plot(trainValidHistory.history['loss'], label='Training loss')
plt.plot(trainValidHistory.history['val_loss'], label='Validation loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(8,5))                                                               # Accuracy curves
plt.plot(trainValidHistory.history['accuracy'], label='Training accuracy')
plt.plot(trainValidHistory.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.imshow(confusionMatrix)                                                             # Confusion matrix
plt.colorbar()
plt.title("Model Confusion Matrix")
plt.xticks([0,1,2,3,4,5,6,7,8,9], ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
plt.yticks([0,1,2,3,4,5,6,7,8,9], ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
for i in range(confusionMatrix.shape[0]):
    for j in range(confusionMatrix.shape[1]):
        plt.text(j,i,confusionMatrix[i,j], ha="center", va="center")
plt.show()

print("Evaluation complete! (Classification metrics)")



'''
Course:  CEG 4195
Name:    Henry Li

Running instructions:
1. Open a command prompt.
2. Enter "conda init"
3. Enter "conda activate <environment name>"
4. Enter "python assignment6.py" to run the code. 

Dataset: ylecun/mnist
'''
#=======================================================
from datasets import load_dataset
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
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

X_train = X_train.reshape(-1,28,28,1)/255.0
X_test = X_test.reshape(-1,28,28,1)/255.0

print("Normalization complete!")

# ===== CNN model =====
model = keras.Sequential([
    layers.Conv2D(32, (3,3), padding='same', input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),

    layers.Dense(10, activation='softmax')
])

# ===== Compile model =====
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

# ===== Train model =====
trainValidHistory = model.fit(X_train, y_train, epochs=20, validation_split=0.2)

# ===== Predictions =====
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# ===== Performance metrics =====
loss, accuracy = model.evaluate(X_test, y_test)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
confusionMatrix = confusion_matrix(y_test, y_pred)

# Numbers:
print(f"Accuracy: {accuracy*100:.2f}%")      # Accuracy  = ~97%
print(f"Loss: {loss:.4f}")                   # Loss      = ~0.1034
print(f"Precision: {precision*100:.2f}%")    # Precision = ~97%
print(f"Recall: {recall*100:.2f}%")          # Recall    = ~97%
print(f"F1-Score: {f1*100:.2f}%")            # F1-Score  = ~97%

# Loss graph: 
plt.figure(figsize=(8,5))                                                              # Loss curves
plt.plot(trainValidHistory.history['loss'], label='Training loss')
plt.plot(trainValidHistory.history['val_loss'], label='Validation loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Accuracy graph: 
plt.figure(figsize=(8,5))                                                               # Accuracy curves
plt.plot(trainValidHistory.history['accuracy'], label='Training accuracy')
plt.plot(trainValidHistory.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Confusion matrix:
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


'''
Course:  CEG 4195
Name:    Henry Li

Running instructions:
1. Open a command prompt.
2. Enter "conda init"
3. Enter "conda activate <environment name>"
4. Enter "python assignment<#>.py" to run the code. 

Dataset: zalando-datasets/fashion_mnist
'''

#=============================================================================
#                              Imports
#=============================================================================
from datasets import load_dataset

import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

#=============================================================================
#                  Constants and function definitions
#=============================================================================
IMG_SIZE = 64
NUM_CLASSES = 10
LABELED_SET_PORTION = 0.2
NUM_DATASET_VALUES = 2000

def ResNetPreprocessing(images):
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    images = tf.image.resize(images, (IMG_SIZE,IMG_SIZE))
    images = tf.image.grayscale_to_rgb(images)
    return images

#=============================================================================
#                   Dataset selection and preprocessing  
#=============================================================================

# ===== Load dataset =====
print("Loading dataset...")
dataset = load_dataset("zalando-datasets/fashion_mnist")
print("Dataset loaded!")

# ===== Extract data =====
print("Extracting data...")

train = dataset["train"].shuffle(seed=42)                        # 42 is for reproducible randomness.
test = dataset["test"].shuffle(seed=42)   

X_trainDataset = train["image"][:NUM_DATASET_VALUES]             # Limit amount of data so it doesn't run too slow.
Y_trainDataset = train["label"][:NUM_DATASET_VALUES]
X_testDataset = test["image"][:NUM_DATASET_VALUES]
Y_testDataset = test["label"][:NUM_DATASET_VALUES]

print("Data extraction complete!")

# ===== Normalize pixel values =====
print("Normalizing pixel values...")

X_train = []                                        # Put all image data into a numpy array.
for image in X_trainDataset:
    X_train.append(np.array(image))
X_train = np.stack(X_train)

X_test = []                                         
for image in X_testDataset:
    X_test.append(np.array(image))
X_test = np.stack(X_test)

Y_train = np.array(Y_trainDataset)            
Y_test = np.array(Y_testDataset)

X_train = X_train.reshape(-1,28,28,1)/255.0         # Normalize to a value between 0 and 1.
X_test = X_test.reshape(-1,28,28,1)/255.0

print("Normalization complete!")

# ===== Resize and change to RGB for ResNet usage =====
print("ResNet preprocessing...")
X_train = ResNetPreprocessing(X_train)
X_test = ResNetPreprocessing(X_test)
print("ResNet preprocessing complete!")

# ===== Splitting dataset =====
print("Splitting dataset...")
# Labeled set
labeledSetSize = int(NUM_DATASET_VALUES*LABELED_SET_PORTION)
X_labeled = X_train[:labeledSetSize]
Y_labeled = Y_train[:labeledSetSize]

# Unlabeled set
X_unlabeled = X_train[labeledSetSize:]
Y_unlabeled = Y_train[labeledSetSize:]

# Test set
# Declared in normalization section

print("Splitting dataset complete!")

# =================================================================================================
#                                      Pipeline design  
# =================================================================================================
# ===== Data Augmentation =====
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    #tf.keras.layers.RandomRotation(0.1),
    #tf.keras.layers.RandomZoom(0.1)
])

# ===== Dealing with inputs =====
input_layer = tf.keras.Input(shape=(IMG_SIZE,IMG_SIZE,3))                                               # Define what the inputs are that enter the first layer.
model_inputs = data_augmentation(input_layer)                                                           # Apply augumentation to the image inputs.

# ===== ResNet CNN =====
baseModel = ResNet50(include_top = False, weights = 'imagenet', input_shape =(IMG_SIZE,IMG_SIZE,3))     # ResNet CNN (without the classification head).
baseModel.trainable = False                                                                             # Don't allow modifying of the pretrained features.

featureMaps = baseModel(model_inputs, training=False)                                                   # Output the feature maps for each image.
featureMaps = GlobalAveragePooling2D()(featureMaps)                                                     # Summarize info of the feature map.
featureMaps = Dense(128, activation='relu')(featureMaps)                                                # Abstract representation of the image.

predictions = Dense(NUM_CLASSES, activation='softmax')(featureMaps)                                     # Produce class probabilities.

model = Model(inputs=input_layer, outputs=predictions)                                                  # Connect the CNN together.
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# ===== Train on the labelled set =====
history = model.fit(X_labeled, Y_labeled, epochs=5, batch_size=32, validation_data=(X_test, Y_test))     

# ===== Unfreeze top layers for training =====
baseModel.trainable = True
for layer in baseModel.layers[:-50]:                                                                  
    layer.trainable = False

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# ===== Iteratively retrain ===== 
for i in range(3):
    pseudoProbabilities = model.predict(X_unlabeled)                                                        # Predict class probabilities on unlabeled set.
    pseudoLabels = np.argmax(pseudoProbabilities, axis = 1)                                                 # Get the predicted label.
    
    confidence = np.max(pseudoProbabilities, axis=1)                                                        # Only keep the highly confident (>90% confidence) labels.
    mask = confidence > 0.9
    X_pseudo = X_unlabeled[mask]
    Y_pseudo = pseudoLabels[mask]

    # Train on a combination of the labelled set and peseudo labelled set 
    X_combined = np.concatenate([X_labeled, X_pseudo], axis = 0)
    Y_combined = np.concatenate([Y_labeled, Y_pseudo], axis = 0)
    historyRetrained = model.fit(X_combined, Y_combined, epochs = 5, batch_size=32, validation_data=(X_test, Y_test))

# =================================================================================================
#                                      Performance metrics  
# =================================================================================================
Y_pred_probs = model.predict(X_test)
Y_preds = np.argmax(Y_pred_probs, axis=1)

# Classification report:
classifReport = classification_report(Y_test, Y_preds)
print(classifReport)

# Loss graph: 
plt.figure(figsize=(8,5))                                                             
plt.plot(historyRetrained.history['loss'], label='Training loss')
plt.plot(historyRetrained.history['val_loss'], label='Validation loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Accuracy graph: 
plt.figure(figsize=(8,5))                                                               
plt.plot(historyRetrained.history['accuracy'], label='Training accuracy')
plt.plot(historyRetrained.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Confusion matrix heatmap:
confusionMatrix = confusion_matrix(Y_test, Y_preds)
plt.imshow(confusionMatrix)                                                             
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
    
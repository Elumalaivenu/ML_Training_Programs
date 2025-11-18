import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
  
# Step 2: Load and Prepare the Data 
# We’ll use the MNIST dataset again, but this time we'll reshape the data for a CNN. 
# Load the dataset 
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() 
 
 
# Normalize the data 
# In short: it standardizes pixel values to improve learning efficiency
x_train = x_train / 255.0 
x_test = x_test / 255.0 
 

# Reshape for CNN input (28x28x1) -> 1 color channel 
# CNNs expect input in 4D format: (samples, height, width, channels).
x_train = x_train.reshape(-1, 28, 28, 1) 
x_test = x_test.reshape(-1, 28, 28, 1) 
  

# Step 3: Build the CNN Model. Now, let’s define the layers of the CNN.
# This layer (Conv2D(32, (3, 3))detects low-level features like edges and textures.
# pooling layer keeps important features while reducing computation and overfitting
# Conv2D(64, (3, 3) Learning more complex features (like shapes or corners) & Deeper layers learn higher-level patterns.
model = keras.Sequential([ 
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), keras.layers.MaxPooling2D((2, 2)), 
    keras.layers.Conv2D(64, (3, 3), activation='relu'),keras.layers.MaxPooling2D((2, 2)),keras.layers.Flatten(), 
    keras.layers.Dense(64, activation='relu'), 
    keras.layers.Dense(10, activation='softmax')  # 10 output classes 
]) 
  
# Step 4: Compile the Model 
# We’ll use Adam optimizer and Sparse Categorical Crossentropy for multi-class classification. 
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
  
# Step 5: Train the Model 
# Let’s train the CNN model on the MNIST data. 
model.fit(x_train, y_train, epochs=5) 
  
# Step 6: Evaluate the Model 
# We’ll check how well the model performs on the test data. 
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy}") 
  
# Step 7: Make Predictions
# Now we can use the trained model to predict some digits.

predictions = model.predict(x_test)

# Predict the first test image
predicted_label = np.argmax(predictions[0])
print(f"Predicted: {predicted_label}")
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predicted_label}")
plt.show()

# Add these lines for comprehensive analysis
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Get all predictions
y_pred = np.argmax(model.predict(x_test), axis=1)

# Classification report
print("\n=== DETAILED METRICS ===")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Digit Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

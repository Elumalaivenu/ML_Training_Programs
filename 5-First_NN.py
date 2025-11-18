import tensorflow as tf
from tensorflow import keras
import numpy as np

# Input data (hours studied)
X = np.array([[1], [2], [3], [4], [5]], dtype=float)

# Output labels (0 = Fail, 1 = Pass)
y = np.array([[0], [0], [0], [1], [1]], dtype=float)

# Build a simple model
model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1], activation='sigmoid')
])
 
# Compile the model 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
 
# Train the model 
model.fit(X, y, epochs=50) 
 
# Test the model 
print(model.predict(np.array([[1.5], [3.5]]))) 

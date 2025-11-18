import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Step 1: Simulated Temperature Data (e.g., daily average)
days = np.arange(0, 365)
temperatures = 20 + 10 * np.sin(2 * np.pi * days / 365) + np.random.normal(0, 0.5, len(days))

# Step 2: Prepare dataset (past 7 days → next day)
window_size = 7
X = np.array([temperatures[i:i+window_size] for i in range(len(temperatures) - window_size)])
Y = np.array([temperatures[i+window_size] for i in range(len(temperatures) - window_size)])

# Reshape for RNN input (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))
Y = Y.reshape((Y.shape[0], 1))

# Step 3: Build SimpleRNN Model
model = keras.Sequential([
    layers.SimpleRNN(32, activation='tanh', input_shape=(window_size, 1)),
    layers.Dense(1)
])

# Step 4: Compile the model
model.compile(optimizer='adam', loss='mse')

# Step 5: Train the model
model.fit(X, Y, epochs=100, batch_size=16, verbose=1)

# Step 6: Make predictions (FIXED - use X not Y)
print("Making predictions...")
predictions = model.predict(X)

# Step 6.1: Calculate performance metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(Y, predictions.flatten())
mae = mean_absolute_error(Y, predictions.flatten())
rmse = np.sqrt(mse)

print(f"\n=== TEMPERATURE FORECASTING PERFORMANCE ===")
print(f"Mean Squared Error (MSE): {mse:.4f} °C²")
print(f"Mean Absolute Error (MAE): {mae:.4f} °C")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f} °C")
print(f"Training samples: {len(X)}")
print(f"Temperature range: {temperatures.min():.1f}°C to {temperatures.max():.1f}°C")

# Step 7: Plot results
plt.figure(figsize=(12, 6))
plt.plot(Y, label='Actual Next-Day Temperature', color='blue', alpha=0.7)
plt.plot(predictions.flatten(), label='Predicted Next-Day Temperature', color='orange', alpha=0.7)
plt.title('SimpleRNN – Next-Day Temperature Forecasting')
plt.xlabel('Days')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Step 8: Show model summary
print("\n=== MODEL ARCHITECTURE ===")
model.summary()


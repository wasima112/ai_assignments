import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


#Step1: Define the model skeleton/Architecture
model = keras.Sequential([
    
    # First hidden layer with 64 neurons. The input layer receives 10 features as input
    layers.Dense(64, activation='relu', input_shape=((10,)) ),
    # second Hidden layer
    layers.Dense(32, activation='relu'),
    #output layer
    layers.Dense(1, activation='sigmoid')
])
print("\n\n\nCODE START\n\n\n") 

print("Model created 10-64-32-1");

# compile the model so it is ready for training
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
)
# Show model Summary
model.summary()

model.save("my_first_model.h5")
print("Model Saved successfully!!!")

# loaded_model = keras.model.load_model("my_first_model.h5")

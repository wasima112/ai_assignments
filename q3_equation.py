import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


def equation_builder(eq_type, n):
    x = np.random.uniform(low=-10, high=10, size=n).reshape(-1, 1)
    
    if eq_type == 1:
        y = 5*x + 10
    elif eq_type == 2:
        y = 3*x**2 + 5*x + 10
    else:
        y = 4*x**3 + 3*x**2 + 5*x + 10
        
    return x, y


def prepare_data(x, y):
    total_n = len(x)
    indices = np.random.permutation(total_n)
    
    x, y = x[indices], y[indices]

    train_percent = int(total_n * 0.7)
    val_percent = int(total_n * 0.1)
    test_percent = total_n - train_percent - val_percent
    
    x_train = x[:train_percent]
    y_train = y[:train_percent]
    
    x_val = x[train_percent: train_percent + val_percent]
    y_val = y[train_percent: train_percent + val_percent]
    
    x_test = x[train_percent + val_percent:]
    y_test = y[train_percent + val_percent:]
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def build_model(x_train):
    normalize_layer = layers.Normalization(axis=-1)
    normalize_layer.adapt(x_train)
    
    model = keras.Sequential([
        normalize_layer,
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']  # Added evaluation metric
    )
    
    return model


def train_and_test_model(model, x_train, y_train, x_val, y_val):
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=16,
        epochs=20,
        verbose=0
    )
    
    return history



def evaluate_and_predict(model, x_test, y_test):
    
    loss, mae = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test)
    
    return loss, mae, predictions


def visualize_results(x_test, predictions, y_test, eq_type):
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_test, y_test, color="blue", label="Actual")
    plt.scatter(x_test, predictions, color='red', label="Predicted")
    plt.title("Results (Actual vs Predicted)")
    plt.legend()
    if eq_type == 1:
        plt.savefig("linear_result.png")
    elif eq_type == 2:
        plt.savefig("quadratic_result.png")
    else:
        plt.savefig("Cubic_result.png")
    print("Result updated")


def main():
    # Change equation type here: 1 (linear), 2 (quadratic), 3 (cubic)
    eq_type = 3
    x, y = equation_builder(eq_type, 10000)
    
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = prepare_data(x, y)
    
    model = build_model(x_train)
    
    model.summary()
    
    # Train model
    history = train_and_test_model(model, x_train, y_train, x_val, y_val)
    
    # Evaluate and predict
    loss, mae, predictions = evaluate_and_predict(model, x_test, y_test)
    
    print("Test Loss (MSE):", loss)
    print("Test MAE:", mae)
    
    # Plot results
    visualize_results(x_test, predictions, y_test, eq_type)


if __name__ == "__main__":
    main()

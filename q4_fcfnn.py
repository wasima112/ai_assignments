import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


BATCH_SIZE = 16
EPOCHS = 1


# -------------------------
# Load Any Dataset
# -------------------------
def load_data(dataset_name):
    print(f"\n\n====== LOADING {dataset_name.upper()} DATASET =====\n\n")
    
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype("float32")
        x_test = x_test.reshape(-1, 28, 28, 1).astype("float32")
        input_shape = (28, 28, 1)

    elif dataset_name == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype("float32")
        x_test = x_test.reshape(-1, 28, 28, 1).astype("float32")
        input_shape = (28, 28, 1)

    elif dataset_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        input_shape = (32, 32, 3)

    else:
        raise ValueError("Invalid dataset name")

    return x_train, y_train, x_test, y_test, input_shape


# -------------------------
# Build Model (Dynamic Input)
# -------------------------
def build_model(input_shape):
    print("\n\n====== Build Model Architecture =====\n\n")
    
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=input_shape),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# -------------------------
# Train & Evaluate
# -------------------------
def train_model(model, x_train, y_train, x_test, y_test):
    print("\n\n====== Train Model =====\n\n")
    
    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )
    
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    return loss, accuracy


# -------------------------
#  Visualization (Handles RGB & Grayscale)
# -------------------------
def visualize_test_results(model, x_test, y_test, num_of_img=5):
    print("\n\n====== Visualizing Results =====\n\n")
    
    indices = np.random.choice(len(x_test), num_of_img)
    images = x_test[indices]
    labels = y_test[indices]
    
    preds = model.predict(images, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    
    plt.figure(figsize=(12, 4))
    
    for i in range(num_of_img):
        plt.subplot(1, num_of_img, i + 1)
        
        if images.shape[-1] == 1:
            plt.imshow(images[i].squeeze(), cmap="gray")
        else:
            plt.imshow(images[i].astype("uint8"))
        
        plt.title(f"A:{labels[i]} | P:{pred_labels[i]}", fontsize=9)
        plt.axis("off")

    plt.savefig("result.png")


# -------------------------
# Main
# -------------------------
def main():

    # CHANGE DATASET HERE:
    # "mnist"
    # "fashion_mnist"
    # "cifar10"
    dataset_name = "cifar10"

    x_train, y_train, x_test, y_test, input_shape = load_data(dataset_name)

    model = build_model(input_shape)
    model.summary()

    loss, accuracy = train_model(model, x_train, y_train, x_test, y_test)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {loss:.4f}")

    visualize_test_results(model, x_test, y_test)


if __name__ == "__main__":
    main()

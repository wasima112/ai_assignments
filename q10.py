# ==========================================================
# 1. IMPORT LIBRARIES
# ==========================================================
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE


# ==========================================================
# 2. LOAD DATA (KERAS MNIST – RAM SAFE)
# ==========================================================
def load_data():

    img_size = (224, 224)
    batch_size = 8   # keep small for Colab

    # Load MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Binary classification: digits 0 vs 1
    train_idx = np.where((y_train == 0) | (y_train == 1))
    test_idx = np.where((y_test == 0) | (y_test == 1))

    x_train, y_train = x_train[train_idx], y_train[train_idx]
    x_test, y_test = x_test[test_idx], y_test[test_idx]

    # Normalize
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Expand channel dimension (28,28,1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Validation split (80/20)
    split = int(0.8 * len(x_train))
    x_val, y_val = x_train[split:], y_train[split:]
    x_train, y_train = x_train[:split], y_train[:split]

    # Create tf.data datasets
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Preprocessing function (batch-wise resize)
    def preprocess(image, label):
        image = tf.image.resize(image, img_size)
        image = tf.image.grayscale_to_rgb(image)
        return image, label

    train_ds = train_ds.map(preprocess).batch(batch_size).prefetch(AUTOTUNE)
    val_ds = val_ds.map(preprocess).batch(batch_size).prefetch(AUTOTUNE)
    test_ds = test_ds.map(preprocess).batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds


# ==========================================================
# 3. CREATE MODEL (TRANSFER LEARNING / FINE-TUNING)
# ==========================================================
def create_model(trainable_layers=None):
    """
    trainable_layers:
        None -> Freeze all VGG16 layers (transfer learning)
        int  -> Fine-tune last N layers
        "all"-> Fine-tune entire VGG16
    """

    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze all layers first
    for layer in base_model.layers:
        layer.trainable = False

    # Fine-tuning logic
    if trainable_layers == "all":
        base_model.trainable = True

    elif isinstance(trainable_layers, int):
        base_model.trainable = True
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False

    # Classification head (RAM optimized)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==========================================================
# 4. TRAIN MODEL
# ==========================================================
def train_model(model, train_ds, val_ds, title):

    print("\n==============================")
    print("Training:", title)
    print("==============================\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=2,   # small for Colab
        verbose=1
    )

    return history


# ==========================================================
# 5. EVALUATE MODEL
# ==========================================================
def evaluate_model(model, test_ds, title):

    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Test Accuracy ({title}): {acc:.4f}")
    return acc


# ==========================================================
# 6. PLOT VALIDATION ACCURACY
# ==========================================================
def plot_histories(histories, labels):

    plt.figure(figsize=(8, 5))

    for h, label in zip(histories, labels):
        plt.plot(h.history["val_accuracy"], marker="o", label=label)

    plt.title("Effect of Fine-Tuning on Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


# ==========================================================
# 7. MAIN FUNCTION
# ==========================================================
def main():

    train_ds, val_ds, test_ds = load_data()

    histories = []
    labels = []

    # A. Transfer Learning (Frozen VGG16)
    model_frozen = create_model(trainable_layers=None)
    h1 = train_model(model_frozen, train_ds, val_ds, "Frozen VGG16")
    evaluate_model(model_frozen, test_ds, "Frozen")
    histories.append(h1)
    labels.append("Frozen")

    # B. Partial Fine-Tuning (Last 4 layers)
    model_partial = create_model(trainable_layers=4)
    h2 = train_model(model_partial, train_ds, val_ds, "Partial Fine-Tuning")
    evaluate_model(model_partial, test_ds, "Partial")
    histories.append(h2)
    labels.append("Partial")

    # C. Full Fine-Tuning (Entire VGG16)
    model_full = create_model(trainable_layers="all")
    h3 = train_model(model_full, train_ds, val_ds, "Full Fine-Tuning")
    evaluate_model(model_full, test_ds, "Full")
    histories.append(h3)
    labels.append("Full")

    plot_histories(histories, labels)


# ==========================================================
# 8. RUN PROGRAM
# ==========================================================
if __name__ == "__main__":
    main()

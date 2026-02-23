# ==========================================================
# 1. IMPORT LIBRARIES
# ==========================================================
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np


# ==========================================================
# 2. LOAD DATA (USING KERAS MNIST)
# ==========================================================
def load_data():

    img_size = (224, 224)
    batch_size = 16

    # Load MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Keep only digits 0 and 1
    train_filter = np.where((y_train == 0) | (y_train == 1))
    test_filter = np.where((y_test == 0) | (y_test == 1))

    x_train, y_train = x_train[train_filter], y_train[train_filter]
    x_test, y_test = x_test[test_filter], y_test[test_filter]

    # Normalize
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Expand channel dimension
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Split validation (80/20)
    val_split = int(0.8 * len(x_train))

    x_val = x_train[val_split:]
    y_val = y_train[val_split:]

    x_train = x_train[:val_split]
    y_train = y_train[:val_split]

    # Create tf.data datasets
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Resize INSIDE pipeline (RAM safe)
    def preprocess(image, label):
        image = tf.image.resize(image, img_size)
        image = tf.image.grayscale_to_rgb(image)
        return image, label

    train_ds = train_ds.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


# ==========================================================
# 3. CREATE MODEL WITH DIFFERENT FINE-TUNING OPTIONS
# ==========================================================
def create_model(trainable_layers=None):

    base_model = VGG16(weights="imagenet", include_top=False,
                       input_shape=(224, 224, 3))

    # Freeze everything first
    for layer in base_model.layers:
        layer.trainable = False

    # ---------- Fine-tuning options ----------
    if trainable_layers == "all":
        base_model.trainable = True

    elif isinstance(trainable_layers, int):
        base_model.trainable = True
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
    # -----------------------------------------

    # Classifier head
    x = base_model.output
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=base_model.input, outputs=x)

    model.compile(
        optimizer="adam",
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
        epochs=1,
        verbose=1
    )

    return history


# ==========================================================
# 5. EVALUATE ON TEST DATA
# ==========================================================
def evaluate_model(model, test_ds, title):

    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"\nTest Accuracy ({title}): {acc:.4f}")
    return acc


# ==========================================================
# 6. PLOT VALIDATION ACCURACY COMPARISON
# ==========================================================
def plot_histories(histories, labels):

    plt.figure()

    for h, label in zip(histories, labels):
        plt.plot(h.history["val_accuracy"], label=label)

    plt.title("Fine-Tuning Comparison (Validation Accuracy)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()


# ==========================================================
# 7. MAIN FUNCTION
# ==========================================================
def main():

    train_ds, val_ds, test_ds = load_data()

    histories = []
    labels = []

    # A. Frozen
    model_frozen = create_model(trainable_layers=None)
    h1 = train_model(model_frozen, train_ds, val_ds, "Frozen VGG16")
    evaluate_model(model_frozen, test_ds, "Frozen")
    histories.append(h1)
    labels.append("Frozen")

    # B. Partial Fine-Tuning
    model_partial = create_model(trainable_layers=4)
    h2 = train_model(model_partial, train_ds, val_ds, "Partial Fine-Tuning")
    evaluate_model(model_partial, test_ds, "Partial")
    histories.append(h2)
    labels.append("Partial")

    # C. Full Fine-Tuning
    model_full = create_model(trainable_layers="all")
    h3 = train_model(model_full, train_ds, val_ds, "Full Fine-Tuning")
    evaluate_model(model_full, test_ds, "Full")
    histories.append(h3)
    labels.append("Full")

    plot_histories(histories, labels)


if __name__ == "__main__":
    main()

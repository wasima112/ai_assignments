# ==========================================================
# 1. IMPORT LIBRARIES
# ==========================================================
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


# ==========================================================
# 2. LOAD DATASET (BEANS)
# ==========================================================
def load_data(img_size=(128, 128)):

    (train_ds, test_ds), ds_info = tfds.load(
        "beans",
        split=["train", "test"],
        as_supervised=True,
        with_info=True
    )

    num_classes = ds_info.features["label"].num_classes

    def preprocess(image, label):
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    train_ds = train_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds, num_classes


# ==========================================================
# 3. BUILD MODEL FUNCTION
# ==========================================================
def build_model(input_shape, num_classes,
                use_augmentation=False,
                use_dropout=False):

    inputs = Input(shape=input_shape)
    x = inputs

    # -------------------------
    # Data Augmentation
    # -------------------------
    if use_augmentation:
        x = layers.RandomRotation(0.2)(x)
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomZoom(0.2)(x)

    # -------------------------
    # Convolution Block 1
    # -------------------------
    x = layers.Conv2D(32, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    # -------------------------
    # Convolution Block 2
    # -------------------------
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    # -------------------------
    # Fully Connected
    # -------------------------
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    if use_dropout:
        x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ==========================================================
# 4. TRAIN MODEL
# ==========================================================
def train_model(model, train_ds, test_ds, epochs=8):

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
        verbose=1
    )

    return history


# ==========================================================
# 5. PLOT RESULTS
# ==========================================================
def plot_results(history_baseline,
                 history_dropout,
                 history_aug_dropout):

    # -------------------------
    # Accuracy Plot
    # -------------------------
    plt.figure()

    plt.plot(history_baseline.history['val_accuracy'],
             label="Baseline")

    plt.plot(history_dropout.history['val_accuracy'],
             label="Dropout")

    plt.plot(history_aug_dropout.history['val_accuracy'],
             label="Aug + Dropout")

    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("overfitting_accuracy.png")
    plt.close()

    # -------------------------
    # Loss Plot
    # -------------------------
    plt.figure()

    plt.plot(history_baseline.history['val_loss'],
             label="Baseline")

    plt.plot(history_dropout.history['val_loss'],
             label="Dropout")

    plt.plot(history_aug_dropout.history['val_loss'],
             label="Aug + Dropout")

    plt.title("Validation Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("overfitting_loss.png")
    plt.close()

    print("Plots saved successfully.")


# ==========================================================
# 6. MAIN FUNCTION
# ==========================================================
def main():

    # Load data
    train_ds, test_ds, num_classes = load_data()

    input_shape = (128, 128, 3)

    # ---------------------------------
    # Baseline Model
    # ---------------------------------
    print("\nTraining Baseline Model")
    model_baseline = build_model(input_shape, num_classes,
                                 use_augmentation=False,
                                 use_dropout=False)
    history_baseline = train_model(model_baseline, train_ds, test_ds)

    # ---------------------------------
    # Dropout Model
    # ---------------------------------
    print("\nTraining Dropout Model")
    model_dropout = build_model(input_shape, num_classes,
                                use_augmentation=False,
                                use_dropout=True)
    history_dropout = train_model(model_dropout, train_ds, test_ds)

    # ---------------------------------
    # 3 Augmentation + Dropout Model
    # ---------------------------------
    print("\nTraining Augmentation + Dropout Model")
    model_aug_dropout = build_model(input_shape, num_classes,
                                    use_augmentation=True,
                                    use_dropout=True)
    history_aug_dropout = train_model(model_aug_dropout,
                                      train_ds,
                                      test_ds)

    # Plot comparison
    plot_results(history_baseline,
                 history_dropout,
                 history_aug_dropout)


# ==========================================================
# 7. RUN
# ==========================================================
if __name__ == "__main__":
    main()

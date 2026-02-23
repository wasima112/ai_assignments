# ==========================================================
# 1. IMPORT LIBRARIES
# ==========================================================
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np


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
# 3. CREATE AUGMENTATION LAYERS
# ==========================================================
def create_augmentation_layers():
    rotation = layers.RandomRotation(0.2)
    flip = layers.RandomFlip("horizontal")
    zoom = layers.RandomZoom(0.2)
    return rotation, flip, zoom


# ==========================================================
# 4. SAVE IMAGE FUNCTION
# ==========================================================
def save_image(image, filename):
    plt.figure()
    plt.imshow(image)
    plt.axis("off")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


# ==========================================================
# 5. GENERATE AND SAVE SAMPLE AUGMENTATIONS
# ==========================================================
def generate_and_save_images(dataset, rotation, flip, zoom):

    for images, labels in dataset.take(1):
        sample = images[0]
        sample_batch = tf.expand_dims(sample, 0)

        rotated = rotation(sample_batch, training=True)
        flipped = flip(sample_batch, training=True)
        zoomed = zoom(sample_batch, training=True)

        save_image(sample.numpy(), "original.png")
        save_image(rotated[0].numpy(), "rotated.png")
        save_image(flipped[0].numpy(), "flipped.png")
        save_image(zoomed[0].numpy(), "zoomed.png")

        break

    print("Augmented images saved.")


# ==========================================================
# 6. BUILD CNN MODEL (FUNCTIONAL API)
# ==========================================================
def build_model(input_shape, num_classes, use_augmentation=False):

    inputs = Input(shape=input_shape)
    x = inputs

    if use_augmentation:
        x = layers.RandomRotation(0.2)(x)
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomZoom(0.2)(x)

    x = layers.Conv2D(32, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ==========================================================
# 7. TRAIN MODEL
# ==========================================================
def train_model(model, train_ds, test_ds):
    history = model.fit(
        train_ds,
        epochs=3,
        validation_data=test_ds,
        verbose=0
    )
    return history


# ==========================================================
# 8. PLOT ACCURACY & LOSS COMPARISON
# ==========================================================
def plot_comparison(history_no_aug, history_aug):

    # Accuracy plot
    plt.figure()
    plt.plot(history_no_aug.history['val_accuracy'], label="No Aug - Val Acc")
    plt.plot(history_aug.history['val_accuracy'], label="Aug - Val Acc")
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("accuracy_comparison.png")
    plt.close()

    # Loss plot
    plt.figure()
    plt.plot(history_no_aug.history['val_loss'], label="No Aug - Val Loss")
    plt.plot(history_aug.history['val_loss'], label="Aug - Val Loss")
    plt.title("Validation Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_comparison.png")
    plt.close()

    print("Accuracy and Loss plots saved.")


# ==========================================================
# 9. MAIN FUNCTION
# ==========================================================
def main():

    # Load dataset
    train_ds, test_ds, num_classes = load_data()

    # Create augmentation layers
    rotation, flip, zoom = create_augmentation_layers()

    # Save sample images
    generate_and_save_images(train_ds, rotation, flip, zoom)

    # Model WITHOUT augmentation
    print("\nTraining WITHOUT Augmentation")
    model_no_aug = build_model((128,128,3), num_classes, use_augmentation=False)
    history_no_aug = train_model(model_no_aug, train_ds, test_ds)

    # Model WITH augmentation
    print("\nTraining WITH Augmentation")
    model_aug = build_model((128,128,3), num_classes, use_augmentation=True)
    history_aug = train_model(model_aug, train_ds, test_ds)

    # Plot comparison
    plot_comparison(history_no_aug, history_aug)


# ==========================================================
# 10. RUN
# ==========================================================
if __name__ == "__main__":
    main()

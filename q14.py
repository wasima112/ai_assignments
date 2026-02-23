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
        label = tf.one_hot(label, num_classes)
        return image, label

    train_ds = train_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds, num_classes


# ==========================================================
# 3. BUILD MODEL (CHANGE ACTIVATION)
# ==========================================================
def build_model(input_shape, num_classes, activation_function):

    inputs = Input(shape=input_shape)

    x = layers.Conv2D(32, (3,3), activation=activation_function)(inputs)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3,3), activation=activation_function)(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation=activation_function)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    return model


# ==========================================================
# 4. TRAIN MODEL
# ==========================================================
def train_model(model, train_ds, test_ds, loss_function, epochs=8):

    model.compile(
        optimizer='adam',
        loss=loss_function,
        metrics=['accuracy']
    )

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        verbose=1
    )

    return history


# ==========================================================
# 5. PLOT ACTIVATION FUNCTION RESULTS
# ==========================================================
def plot_activation_results(history_relu, history_sigmoid):

    # ReLU Plot
    plt.figure()
    plt.plot(history_relu.history['accuracy'], label='Train Acc')
    plt.plot(history_relu.history['val_accuracy'], label='Val Acc')
    plt.title("ReLU Activation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("relu_activation_output.png")
    plt.close()

    # Sigmoid Plot
    plt.figure()
    plt.plot(history_sigmoid.history['accuracy'], label='Train Acc')
    plt.plot(history_sigmoid.history['val_accuracy'], label='Val Acc')
    plt.title("Sigmoid Activation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("sigmoid_activation_output.png")
    plt.close()


# ==========================================================
# 6. PLOT LOSS FUNCTION RESULTS
# ==========================================================
def plot_loss_results(history_ce, history_mse):

    # Crossentropy
    plt.figure()
    plt.plot(history_ce.history['accuracy'], label='Train Acc')
    plt.plot(history_ce.history['val_accuracy'], label='Val Acc')
    plt.title("Categorical Crossentropy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("crossentropy_output.png")
    plt.close()

    # MSE
    plt.figure()
    plt.plot(history_mse.history['accuracy'], label='Train Acc')
    plt.plot(history_mse.history['val_accuracy'], label='Val Acc')
    plt.title("Mean Squared Error Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("mse_output.png")
    plt.close()


# ==========================================================
# 7. MAIN FUNCTION
# ==========================================================
def main():

    train_ds, test_ds, num_classes = load_data()
    input_shape = (128, 128, 3)

    # ==================================================
    # PART 1: Activation Function Comparison
    # ==================================================
    print("\nTraining with ReLU Activation")
    model_relu = build_model(input_shape, num_classes, activation_function='relu')
    history_relu = train_model(
        model_relu,
        train_ds,
        test_ds,
        loss_function='categorical_crossentropy'
    )

    print("\nTraining with Sigmoid Activation")
    model_sigmoid = build_model(input_shape, num_classes, activation_function='sigmoid')
    history_sigmoid = train_model(
        model_sigmoid,
        train_ds,
        test_ds,
        loss_function='categorical_crossentropy'
    )

    plot_activation_results(history_relu, history_sigmoid)

    # ==================================================
    # PART 2: Loss Function Comparison
    # ==================================================
    print("\nTraining with Categorical Crossentropy")
    model_ce = build_model(input_shape, num_classes, activation_function='relu')
    history_ce = train_model(
        model_ce,
        train_ds,
        test_ds,
        loss_function='categorical_crossentropy'
    )

    print("\nTraining with Mean Squared Error")
    model_mse = build_model(input_shape, num_classes, activation_function='relu')
    history_mse = train_model(
        model_mse,
        train_ds,
        test_ds,
        loss_function='mse'
    )

    plot_loss_results(history_ce, history_mse)


# ==========================================================
# 8. RUN
# ==========================================================
if __name__ == "__main__":
    main()

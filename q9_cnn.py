import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobile_preprocess
from tensorflow.keras.models import Model


# -------------------------------------------------
# Load and Prepare Image
# -------------------------------------------------
def load_image(path, preprocess_function):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_function(img)
    return img


# -------------------------------------------------
#  Create Feature Extraction Model (Functional API)
# -------------------------------------------------
def create_feature_model(base_model, layer_names):
    
    # Get outputs of selected layers
    outputs = [base_model.get_layer(name).output for name in layer_names]
    
    # Functional model
    feature_model = Model(inputs=base_model.input, outputs=outputs)
    
    return feature_model


# -------------------------------------------------
# Display Feature Maps
# -------------------------------------------------
def show_feature_maps(feature_maps, model_name):
    
    for layer_index, layer_map in enumerate(feature_maps):
        
        num_filters = layer_map.shape[-1]
        size = layer_map.shape[1]
        
        print(f"{model_name} - Layer {layer_index+1} Shape:", layer_map.shape)

        # Show first 8 filters only (for clarity)
        plt.figure(figsize=(12, 6))
        
        for i in range(8):
            plt.subplot(2, 4, i+1)
            plt.imshow(layer_map[0, :, :, i], cmap='gray')
            plt.axis("off")
        
        plt.suptitle(f"{model_name} - Feature Maps Layer {layer_index+1}")
        plt.savefig("feature_maps.png")


# -------------------------------------------------
#  Process One Model
# -------------------------------------------------
def process_model(model_name, model_class, preprocess_function, layer_names, image_path):

    print(f"\n\n===== Processing {model_name} =====\n")

    # Load pretrained model
    base_model = model_class(weights="imagenet", include_top=True)

    # Create feature extraction model
    feature_model = create_feature_model(base_model, layer_names)

    # Load image
    img = load_image(image_path, preprocess_function)

    # Get feature maps
    feature_maps = feature_model.predict(img)

    # Show maps
    show_feature_maps(feature_maps, model_name)


# -------------------------------------------------
# Main Function
# -------------------------------------------------
def main():

    image_path = "cat.jpg"   # Change to your favorite image

    # ---------- VGG16 ----------
    process_model(
        model_name="VGG16",
        model_class=VGG16,
        preprocess_function=vgg_preprocess,
        layer_names=["block1_conv1", "block3_conv1", "block5_conv1"],
        image_path=image_path
    )

    # ---------- ResNet50 ----------
    process_model(
        model_name="ResNet50",
        model_class=ResNet50,
        preprocess_function=resnet_preprocess,
        layer_names=["conv1_conv", "conv3_block1_out", "conv5_block1_out"],
        image_path=image_path
    )

    # ---------- MobileNetV2 ----------
    process_model(
        model_name="MobileNetV2",
        model_class=MobileNetV2,
        preprocess_function=mobile_preprocess,
        layer_names=["Conv1", "block_6_expand", "block_13_expand"],
        image_path=image_path
    )


# Run
if __name__ == "__main__":
    main()

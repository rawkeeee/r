import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import Adam

# Load your model (change path as needed)
@st.cache_resource
def load_model():
    model = build_optimized_cnn_with_vgg16_resnet()
    model.load_weights('/kaggle/working/hybrid_model.h5')  # Replace with your model path
    return model

# Define IMG_SIZE globally
IMG_SIZE = 224

# Vanilla CNN

def build_vanilla_cnn():
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(15, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Optimized CNN

def build_optimized_cnn():
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.3),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.3),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(15, activation='softmax')  # Multi-class
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Optimized CNN + VGG16 + ResNet50 (Hybrid)
def build_optimized_cnn_with_vgg16_resnet():
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))

    for layer in vgg.layers:
        layer.trainable = False
    for layer in resnet.layers:
        layer.trainable = False

    vgg_out = GlobalAveragePooling2D()(vgg.output)
    resnet_out = GlobalAveragePooling2D()(resnet.output)

    merged = Concatenate()([vgg_out, resnet_out])
    x = Dense(256, activation='relu')(merged)
    x = Dropout(0.5)(x)
    output = Dense(15, activation='softmax')(x)

    model = Model(inputs=[vgg.input, resnet.input], outputs=output)
    model.compile(optimizer=Adam(1e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_model()

st.title("Lung Disease Detection from Chest X-rays")
st.write("Upload a chest X-ray image to predict lung condition and view heatmap (Grad-CAM).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    input_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(input_array)[0]
    class_index = np.argmax(pred)
    class_labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
                    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
                    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding']
    result = class_labels[class_index]

    st.markdown(f"### Prediction: `{result}` ({pred[class_index]:.2f})")

    # Grad-CAM Implementation
    def get_gradcam_heatmap(model, image_array, class_index, last_conv_layer="block5_conv3"):
        grad_model = tf.keras.models.Model([model.inputs], [
            model.get_layer(last_conv_layer).output, model.output])

        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(image_array)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_output)[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        conv_output = conv_output[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / np.max(heatmap + 1e-10)
        return heatmap

    heatmap = get_gradcam_heatmap(model, input_array, class_index)
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + np.array(image)

    st.image(np.array(image), caption="Original Image", use_column_width=True)
    st.image(superimposed_img.astype(np.uint8), caption="Grad-CAM", use_column_width=True)

    st.info(f"**Detected Disease Information:**\n\nPredicted: {result}\n\nConfidence: {pred[class_index]:.2f}")

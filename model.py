import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LeakyReLU

import numpy as np
import matplotlib.pyplot as plt

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("Using GPU:", gpus)
else:
    print("GPU not found. Using CPU.")

# Load dataset and preprocess
def load_dataset(dataset_path):
    image_paths = []
    ages = []

    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            age = int(filename.split("_")[0])  # Extract age from filename
            ages.append(age)
            image_paths.append(os.path.join(dataset_path, filename))

    return image_paths, ages

# Preprocess images using TensorFlow pipelines
def preprocess_image(image_path, target_size=(224, 224)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0  # Normalize and convert to float32
    return img

def preprocess_dataset(image_paths, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda path, label: (preprocess_image(path), tf.cast(label, tf.float32)),
                          num_parallel_calls=tf.data.AUTOTUNE)
    #tf.data.AUTOTUNE е специјална константа во TensorFlow која автоматски одредува оптималниот
    # број на паралелни операции што треба да се извршат за време на обработката на податоците.
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Paths and preprocessing
dataset_path = "UTKFace"
image_paths, ages = load_dataset(dataset_path)

# Split dataset
X_train_paths, X_temp_paths, y_train, y_temp = train_test_split(image_paths, ages, test_size=0.2, random_state=42)
X_val_paths, X_test_paths, y_val, y_test = train_test_split(X_temp_paths, y_temp, test_size=0.5, random_state=42)

# Preprocess datasets
train_dataset = preprocess_dataset(X_train_paths, y_train, batch_size=32)
val_dataset = preprocess_dataset(X_val_paths, y_val, batch_size=32)
test_dataset = preprocess_dataset(X_test_paths, y_test, batch_size=32)

# Load pre-trained VGG16 and modify for regression
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze the convolutional base
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for regression

x = Flatten()(base_model.output)
x = Dense(512)(x)
x = LeakyReLU(alpha=0.01)(x)
x = Dropout(0.5)(x)
x = Dense(256)(x)
x = LeakyReLU(alpha=0.01)(x)
x = Dropout(0.5)(x)
x = Dense(128)(x)
x = LeakyReLU(alpha=0.01)(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="linear")(x)


# Define the model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.00016), loss="mean_squared_error", metrics=["mae"])

# Train the model with different epochs and visualize results
def train_and_evaluate_model(epochs_list):
    history_dict = {}

    for epochs in epochs_list:
        print(f"Training with {epochs} epochs...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            verbose=1
        )

        history_dict[epochs] = history

    return history_dict

# Define epochs to test
epochs_list = [24]
history_dict = train_and_evaluate_model(epochs_list)

# Plot results
def plot_statistics(history_dict):
    plt.figure(figsize=(12, 8))

    for epochs, history in history_dict.items():
        plt.plot(history.history['val_mae'], label=f'{epochs} epochs (Val MAE)')
        plt.plot(history.history['mae'], label=f'{epochs} epochs (Train MAE)')

    plt.title('Model Performance with Different Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.show()

plot_statistics(history_dict)

# Evaluate the model on test set
loss, mae = model.evaluate(test_dataset, verbose=1)
print(f"Test Loss: {loss}, Mean Absolute Error: {mae}")

# Save the model
#model.save("vgg16_age_recognition_tf.h5")
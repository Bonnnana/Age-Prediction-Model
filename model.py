import os
import itertools
import json
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
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

# Define hyperparameter space
param_grid = {
    "learning_rate": [1e-2, 1e-3, 1e-4],
    "dense_units": [[256, 128], [512, 256]],
    "dropout_rate": [0.3, 0.5],
    "epochs": [20, 30, 40],
}

# Generate all combinations of parameters
param_combinations = list(itertools.product(
    param_grid["learning_rate"],
    param_grid["dense_units"],
    param_grid["dropout_rate"],
    param_grid["epochs"]
))

# Function to create a model
def create_model(dense_units, dropout_rate, learning_rate):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    for units in dense_units:
        x = Dense(units, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
    output = Dense(1, activation="linear")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error", metrics=["mae"])
    return model

# Train models and record results
results = []
best_model = None
best_model_name = None

for i, (learning_rate, dense_units, dropout_rate, epochs) in enumerate(param_combinations):
    model_name = f"model_{i+1}_lr{learning_rate}_units{'-'.join(map(str, dense_units))}_dropout{dropout_rate}_epochs{epochs}"
    print(f"Training {model_name}...")

    model = create_model(dense_units, dropout_rate, learning_rate)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        verbose=1
    )

    # Evaluate on test set
    loss, mae = model.evaluate(test_dataset, verbose=1)
    print(f"{model_name} - Test Loss: {loss}, Mean Absolute Error: {mae}")

    # Save training history as plot
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f"{model_name} Performance")
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.savefig(f"{model_name}_performance.png")
    plt.close()

    # Save model
    model.save(f"{model_name}.h5")

    # Record results
    results.append({
        "model_id": i+1,
        "model_name": model_name,
        "parameters": {
            "learning_rate": learning_rate,
            "dense_units": dense_units,
            "dropout_rate": dropout_rate,
            "epochs": epochs
        },
        "test_loss": loss,
        "test_mae": mae,
        "history": history.history
    })

    # Track the best model
    if best_model is None or mae < best_model["test_mae"]:
        best_model = {
            "model": model,
            "model_name": model_name,
            "parameters": {
                "learning_rate": learning_rate,
                "dense_units": dense_units,
                "dropout_rate": dropout_rate,
                "epochs": epochs
            },
            "test_loss": loss,
            "test_mae": mae
        }
        best_model_name = model_name

# Save results to a JSON file
with open("grid_search_results.json", "w") as f:
    json.dump(results, f)

# Save the best model
if best_model:
    best_model["model"].save(f"best_model_{best_model_name}.h5")
    print(f"Best model saved as best_model_{best_model_name}.h5")

# Plot summary of all models
def plot_summary(results):
    plt.figure(figsize=(12, 6))
    for result in results:
        history = result["history"]
        label = result["model_name"]
        plt.plot(history['val_mae'], label=f'{label} (Validation)')
        plt.plot(history['mae'], label=f'{label} (Training)')
    plt.title('Summary of All Models')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend(loc='best', fontsize='small')
    plt.savefig("grid_search_summary.png")
    plt.show()

plot_summary(results)

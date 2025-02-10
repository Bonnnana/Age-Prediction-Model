import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_dataset(dataset_path):
    image_paths = []
    ages = []

    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            age = int(filename.split("_")[0])
            ages.append(age)
            image_paths.append(os.path.join(dataset_path, filename))

    return image_paths, ages


def preprocess_image(image_path, target_size=(224, 224)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def preprocess_dataset(image_paths, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda path, label: (preprocess_image(path), tf.cast(label, tf.float32)),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def plot_statistics(history, num_epochs):
    plt.figure(figsize=(12, 8))

    plt.plot(history.history['mae'], label=f'{num_epochs} epochs (Train MAE)')
    plt.plot(history.history['val_mae'], label=f'{num_epochs} epochs (Val MAE)')

    plt.title('Model Performance')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()

    plt.savefig("model_performance.png")
    plt.show()


if __name__ == '__main__':

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("Using GPU:", gpus)
    else:
        print("GPU not found. Using CPU.")

    dataset_path = "UTKFace"
    image_paths, ages = load_dataset(dataset_path)

    X_train_paths, X_temp_paths, y_train, y_temp = train_test_split(image_paths, ages, test_size=0.2, random_state=42)
    X_val_paths, X_test_paths, y_val, y_test = train_test_split(X_temp_paths, y_temp, test_size=0.5, random_state=42)

    train_dataset = preprocess_dataset(X_train_paths, y_train, batch_size=32)
    val_dataset = preprocess_dataset(X_val_paths, y_val, batch_size=32)
    test_dataset = preprocess_dataset(X_test_paths, y_test, batch_size=32)

    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    # Freezing all layers except the last 4
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    # Unfreezing the last 4 layers
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    # Model
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="linear")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer=Adam(learning_rate=1e-5), loss="mean_squared_error", metrics=["mae"])
    num_epochs = 20

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        verbose=1
    )

    plot_statistics(history, num_epochs)
    loss, mae = model.evaluate(test_dataset, verbose=1)
    print(f"Test Loss: {loss}, Mean Absolute Error: {mae}")

    # Save the model
    model.save("vgg16_age_recognition_final_256_128_4_20.h5")

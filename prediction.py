import tensorflow as tf
from sklearn.metrics import r2_score
import numpy as np
from model import preprocess_dataset, load_dataset
from sklearn.model_selection import train_test_split


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")
    return model


def predict_and_evaluate(model, test_dataset, y_test):

    y_pred = model.predict(test_dataset)
    y_pred = np.squeeze(y_pred)

    r2 = r2_score(y_test, y_pred)
    print(f"R² Score: {r2}")
    return r2


if __name__ == "__main__":
    model_path = "vgg16_age_recognition_finetuned.h5"

    dataset_path = "UTKFace"
    image_paths, ages = load_dataset(dataset_path)

    X_train_paths, X_temp_paths, y_train, y_temp = train_test_split(image_paths, ages, test_size=0.2, random_state=42)
    X_val_paths, X_test_paths, y_val, y_test = train_test_split(X_temp_paths, y_temp, test_size=0.5, random_state=42)

    test_dataset = preprocess_dataset(X_test_paths, y_test, batch_size=32)

    model = load_model(model_path)

    r2 = predict_and_evaluate(model, test_dataset, y_test)

    #Output: R² Score: 0.8140280246734619

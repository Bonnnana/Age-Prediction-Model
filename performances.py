import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import tensorflow as tf
from keras.losses import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from model import load_dataset, preprocess_dataset


def load_model(model_path):
    m = tf.keras.models.load_model(model_path)
    print(f"Моделот е вчитан од: {model_path}")
    return m


def evaluate_metrics(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"R² Score: {r2}")
    print(f"MAE: {mae}")


def evaluate_by_age_groups(y_test, y_pred):
    age_groups = {
        "0-5": (0, 5),
        "6-10": (6, 10),
        "11-20": (11, 20),
        "21-30": (21, 30),
        "31-40": (31, 40),
        "41-50": (41, 50),
        "51-60": (51, 60),
        "61+": (61, 100),
    }

    mae_by_group = defaultdict(list)
    count_by_group = defaultdict(int)

    for true_age, pred_age in zip(y_test, y_pred):
        for group, (min_age, max_age) in age_groups.items():
            if min_age <= true_age <= max_age:
                mae_by_group[group].append(abs(true_age - pred_age))
                count_by_group[group] += 1
                break

    mae_results = {group: np.mean(errors) if errors else 0 for group, errors in mae_by_group.items()}

    for group in age_groups.keys():
        num_samples = count_by_group[group]
        mae = mae_results[group]
        print(f"MAE за група {group}: {mae:.2f} (Број на слики: {num_samples})")

    plt.figure(figsize=(10, 5))
    plt.bar(mae_results.keys(), mae_results.values(), color='skyblue')
    plt.xlabel("Возрасна група")
    plt.ylabel("MAE")
    plt.title("Средна апсолутна грешка (MAE) по возрасни групи")
    plt.xticks(rotation=45)
    plt.show()


def evaluate_by_ethnicity(y_test, y_pred, image_paths):
    ethnicity_groups = {
        "0": "Белци",
        "1": "Црнци",
        "2": "Азијци",
        "3": "Индијци",
        "4": "Други",
    }

    mae_by_group = defaultdict(list)
    count_by_group = defaultdict(int)

    # Extracting ethnicity from image format
    for true_age, pred_age, path in zip(y_test, y_pred, image_paths):
        ethnicity = path.split("_")[1]
        if ethnicity in ethnicity_groups:
            mae_by_group[ethnicity_groups[ethnicity]].append(abs(true_age - pred_age))
            count_by_group[ethnicity_groups[ethnicity]] += 1

    mae_results = {group: np.mean(errors) if errors else 0 for group, errors in mae_by_group.items()}

    for group in ethnicity_groups.values():
        if group in mae_results:  # Проверете дали групата постои во mae_results
            num_samples = count_by_group[group]
            mae = mae_results[group]
            print(f"MAE за етничка група {group}: {mae:.2f} (Број на слики: {num_samples})")
        else:
            print(f"Нема податоци за етничка група {group}.")

    plt.figure(figsize=(10, 5))
    plt.bar(mae_results.keys(), mae_results.values(), color='lightgreen')
    plt.xlabel("Етничка група")
    plt.ylabel("MAE")
    plt.title("Средна апсолутна грешка (MAE) по етнички групи")
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    model_path = "vgg16_age_recognition_final_256_128_4_20.h5"
    dataset_path = "UTKFace"

    image_paths, ages = load_dataset(dataset_path)

    X_train_paths, X_temp_paths, y_train, y_temp = train_test_split(image_paths, ages, test_size=0.2, random_state=42)
    X_val_paths, X_test_paths, y_val, y_test = train_test_split(X_temp_paths, y_temp, test_size=0.5, random_state=42)

    test_dataset = preprocess_dataset(X_test_paths, y_test, batch_size=32)

    model = load_model(model_path)
    predictions = model.predict(test_dataset)
    predictions = np.squeeze(predictions)

    evaluate_metrics(y_test, predictions)
    evaluate_by_age_groups(y_test, predictions)
    evaluate_by_ethnicity(y_test, predictions, X_test_paths)




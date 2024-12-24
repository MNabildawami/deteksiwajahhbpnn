import pandas as pd
from sklearn.model_selection import train_test_split
import os

input_csv_path = "output/face_features_dlib.csv"
data = pd.read_csv(input_csv_path)

split_output_folder = "output"
os.makedirs(split_output_folder, exist_ok=True)

def split_and_save_data(data, test_size, ratio_name):
    X = data.drop(columns=["Label"])
    y = data["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_csv_path = os.path.join(split_output_folder, f"data_latih_{ratio_name}.csv")
    test_csv_path = os.path.join(split_output_folder, f"data_uji_{ratio_name}.csv")

    train_data.to_csv(train_csv_path, index=False)
    test_data.to_csv(test_csv_path, index=False)

    print(f"Data latih ({ratio_name}) disimpan di: {train_csv_path}")
    print(f"Data uji ({ratio_name}) disimpan di: {test_csv_path}")

split_and_save_data(data, test_size=0.3, ratio_name="70_30")
split_and_save_data(data, test_size=0.2, ratio_name="80_20")
split_and_save_data(data, test_size=0.1, ratio_name="90_10")

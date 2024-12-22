import os
import random
import pandas as pd
import cv2
import dlib
import numpy as np

# Inisialisasi deteksi wajah dan landmark dengan dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks (2).dat")

# Tentukan path folder yang berisi gambar-gambar dalam Google Drive
image_directory = "footage"

# Buat folder output jika belum ada
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Fungsi untuk ekstraksi landmark wajah
def extract_landmarks(image_path):
    """Ekstraksi landmark wajah dari gambar"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"File {image_path} tidak ditemukan atau tidak dapat dibaca.")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = detector(gray)
    landmarks = []


    for face in faces:
        # Mendapatkan landmark wajah
        shape = predictor(gray, face)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    return landmarks

# Fungsi untuk menghitung fitur geometris berdasarkan landmark
def calculate_geometrical_features(landmarks):
    """Menghitung fitur geometris berdasarkan landmark"""
    features = []

    # Jarak antara mata kiri dan mata kanan (titik 36-45)
    left_eye = np.array(landmarks[36:42])
    right_eye = np.array(landmarks[42:48])

    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)
    eye_distance = np.linalg.norm(left_eye_center - right_eye_center)

    # Jarak antara hidung dan mulut (titik 27-48)
    nose = np.array(landmarks[27:36])
    mouth = np.array(landmarks[48:60])
    nose_center = np.mean(nose, axis=0)
    mouth_center = np.mean(mouth, axis=0)
    nose_mouth_distance = np.linalg.norm(nose_center - mouth_center)

    # Jarak antara kedua alis (titik 17-26)
    left_brow = np.array(landmarks[17:22])
    right_brow = np.array(landmarks[22:27])

    left_brow_center = np.mean(left_brow, axis=0)
    right_brow_center = np.mean(right_brow, axis=0)
    brow_distance = np.linalg.norm(left_brow_center - right_brow_center)

    # Jarak antara hidung dan dagu (titik 27-8)
    chin = np.array(landmarks[5:8])
    nose_chin_distance = np.linalg.norm(nose_center - chin.mean(axis=0))

    # Jarak antara sudut mulut (titik 48 dan 54)
    left_corner_of_mouth = np.array(landmarks[48])
    right_corner_of_mouth = np.array(landmarks[54])
    mouth_corners_distance = np.linalg.norm(left_corner_of_mouth - right_corner_of_mouth)

    # Menambahkan fitur ke dalam daftar
    features.extend([eye_distance, nose_mouth_distance, brow_distance, nose_chin_distance, mouth_corners_distance])

    return features

# Fungsi untuk membagi gambar menjadi data latih dan data uji
def split_images_into_train_test(image_directory, train_size=7, test_size=3):
    all_train_images = []
    all_train_labels = []
    all_train_features = []

    all_test_images = []
    all_test_labels = []
    all_test_features = []

    # Menelusuri semua sub-folder dalam folder image_directory
    for subdir, _, files in os.walk(image_directory):
        # Ambil nama sub-folder untuk digunakan sebagai label
        subfolder_name = os.path.basename(subdir)

        # Mengambil file gambar yang ada dalam sub-folder
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        # Membagi gambar menjadi data latih dan data uji (7 untuk latih dan 3 untuk uji)
        random.shuffle(image_files)

        train_images = image_files[:train_size]
        test_images = image_files[train_size:train_size + test_size]

        # Menambahkan gambar latih dan fitur ke dalam daftar data latih
        for image in train_images:
            image_path = os.path.join(subdir, image)
            landmarks = extract_landmarks(image_path)

            if landmarks:
                features = calculate_geometrical_features(landmarks)
                all_train_images.append(image)
                all_train_labels.append(subfolder_name)
                all_train_features.append(features)

        # Menambahkan gambar uji dan fitur ke dalam daftar data uji
        for image in test_images:
            image_path = os.path.join(subdir, image)
            landmarks = extract_landmarks(image_path)

            if landmarks:
                features = calculate_geometrical_features(landmarks)
                all_test_images.append(image)
                all_test_labels.append(subfolder_name)
                all_test_features.append(features)

    return all_train_images, all_train_labels, all_train_features, all_test_images, all_test_labels, all_test_features

# Ambil gambar dari sub-folder, bagi menjadi data latih dan data uji, serta ekstraksi fitur wajah
train_images, train_labels, train_features, test_images, test_labels, test_features = split_images_into_train_test(image_directory)

# Simpan data latih dalam DataFrame dan kemudian simpan ke CSV di folder output
train_df = pd.DataFrame(train_features, columns=["Eye Distance", "Nose-Mouth Distance", "Brow Distance", "Nose-Chin Distance", "Mouth Corners Distance"])
train_df['Label'] = train_labels
train_csv_path = os.path.join(output_folder, "data_latih.csv")
train_df.to_csv(train_csv_path, index=False)

# Simpan data uji dalam DataFrame dan kemudian simpan ke CSV di folder output
test_df = pd.DataFrame(test_features, columns=["Eye Distance", "Nose-Mouth Distance", "Brow Distance", "Nose-Chin Distance", "Mouth Corners Distance"])
test_df['Label'] = test_labels
test_csv_path = os.path.join(output_folder, "data_uji.csv")
test_df.to_csv(test_csv_path, index=False)

print(f"File data latih disimpan di: {train_csv_path}")
print(f"File data uji disimpan di: {test_csv_path}")

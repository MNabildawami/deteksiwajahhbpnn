import os
import cv2
import dlib
import pandas as pd
import numpy as np

# Inisialisasi deteksi wajah dan landmark dengan dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks (2).dat")

def extract_landmarks(image_path):
    """Ekstraksi landmark wajah dari gambar"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"File {image_path} tidak ditemukan atau tidak dapat dibaca.")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        print(f"Tidak ada wajah terdeteksi di {image_path}")
        return []

    # Ambil landmark dari wajah pertama
    shape = predictor(gray, faces[0])
    landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]  # 68 titik landmark

    return landmarks

def calculate_geometrical_features(landmarks):
    """Menghitung fitur geometris berdasarkan landmark"""
    try:
        features = []

        # Jarak antar mata kiri dan mata kanan (titik 36-45)
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
    except Exception as e:
        print(f"Error saat menghitung fitur geometris: {e}")
        return [np.nan] * 5

def process_images_in_directory(directory):
    """Memproses gambar dalam direktori dan sub-direktori"""
    all_features = []
    all_labels = []

    for subdir, _, files in os.walk(directory):
        subfolder_name = os.path.basename(subdir)
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(subdir, file)
                landmarks = extract_landmarks(image_path)

                if landmarks:
                    features = calculate_geometrical_features(landmarks)
                    all_features.append(features)
                    all_labels.append(subfolder_name)
                else:
                    all_features.append([np.nan] * 5)
                    all_labels.append(subfolder_name)

    return all_features, all_labels

# Tentukan folder yang berisi gambar wajah
image_directory = "footage"

# Ekstraksi fitur dari gambar dalam folder
features, labels = process_images_in_directory(image_directory)

# Simpan ke DataFrame
df = pd.DataFrame(features, columns=["Eye Distance", "Nose-Mouth Distance", "Brow Distance", "Nose-Chin Distance", "Mouth Corners Distance"])
df["Label"] = labels

# Buat folder output jika belum ada
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Simpan file ke folder output
csv_path = os.path.join(output_folder, "face_features_dlib.csv")
excel_path = os.path.join(output_folder, "face_features_dlib.xlsx")

df.to_csv(csv_path, index=False)
df.to_excel(excel_path, index=False)

print(f"File CSV disimpan di: {csv_path}")
print(f"File Excel disimpan di: {excel_path}")

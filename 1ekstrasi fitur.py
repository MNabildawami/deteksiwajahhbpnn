import os
import cv2
import dlib
import pandas as pd
import numpy as np

# Inisialisasi deteksi wajah dan landmark dengan dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks (2).dat")

# Preprocessing: Normalisasi pencahayaan
def normalize_lighting(image):
    """Normalisasi pencahayaan menggunakan histogram equalization"""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    normalized_image = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(normalized_image, cv2.COLOR_YCrCb2BGR)

# Preprocessing: Crop wajah dari gambar
def crop_face(image, face):
    """Crop gambar hanya pada area wajah"""
    x, y, w, h = (face.left(), face.top(), face.width(), face.height())
    return image[y:y+h, x:x+w]

def extract_landmarks(image_path):
    """Ekstraksi landmark wajah dari gambar"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"File {image_path} tidak ditemukan atau tidak dapat dibaca.")
        return []

    # Preprocessing: Normalisasi pencahayaan
    img = normalize_lighting(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if len(faces) == 0:
        print(f"Tidak ada wajah terdeteksi di {image_path}")
        return []

    # Ambil wajah pertama
    face = faces[0]

    # Preprocessing: Crop wajah
    cropped_face = crop_face(img, face)

    # Ekstraksi landmark dari wajah pertama
    shape = predictor(gray, face)
    landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

    return landmarks

def calculate_geometrical_features(landmarks):
    """Menghitung fitur geometris berdasarkan landmark"""
    try:
        features = []

        # Eye Distance (Jarak antar mata)
        left_eye = np.array(landmarks[36:42])
        right_eye = np.array(landmarks[42:48])
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        eye_distance = np.linalg.norm(left_eye_center - right_eye_center)

        # Nose-Mouth Distance (Jarak hidung ke mulut)
        nose_center = np.mean(np.array(landmarks[27:36]), axis=0)
        mouth_center = np.mean(np.array(landmarks[48:60]), axis=0)
        nose_mouth_distance = np.linalg.norm(nose_center - mouth_center)

        # Brow Distance (Jarak antar alis)
        left_brow_center = np.mean(np.array(landmarks[17:22]), axis=0)
        right_brow_center = np.mean(np.array(landmarks[22:27]), axis=0)
        brow_distance = np.linalg.norm(left_brow_center - right_brow_center)

        # Nose-Chin Distance (Jarak hidung ke dagu)
        chin = np.array(landmarks[8])
        nose_chin_distance = np.linalg.norm(nose_center - chin)

        # Mouth Corners Distance (Jarak antar sudut mulut)
        left_corner_of_mouth = np.array(landmarks[48])
        right_corner_of_mouth = np.array(landmarks[54])
        mouth_corners_distance = np.linalg.norm(left_corner_of_mouth - right_corner_of_mouth)

        # Eye Aspect Ratio (EAR)
        def eye_aspect_ratio(eye_points):
            return (np.linalg.norm(eye_points[1] - eye_points[5]) + np.linalg.norm(eye_points[2] - eye_points[4])) / (2.0 * np.linalg.norm(eye_points[0] - eye_points[3]))

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Mouth Aspect Ratio (MAR)
        mar = (np.linalg.norm(np.array(landmarks[51]) - np.array(landmarks[57])) +
               np.linalg.norm(np.array(landmarks[52]) - np.array(landmarks[56])) +
               np.linalg.norm(np.array(landmarks[53]) - np.array(landmarks[55]))) / (3.0 * np.linalg.norm(left_corner_of_mouth - right_corner_of_mouth))

        # Face Symmetry
        symmetry = np.mean([np.linalg.norm(np.array(landmarks[i]) - np.array(landmarks[16-i])) for i in range(8)])

        # Feature Angles: Angle between eyes, nose, and chin
        eye_to_nose = nose_center - left_eye_center
        nose_to_chin = chin - nose_center
        eye_nose_chin_angle = np.degrees(np.arccos(
            np.dot(eye_to_nose, nose_to_chin) / (np.linalg.norm(eye_to_nose) * np.linalg.norm(nose_to_chin))
        ))

        # Add features to the list
        features.extend([eye_distance, nose_mouth_distance, brow_distance, nose_chin_distance,
                         mouth_corners_distance, avg_ear, mar, symmetry, eye_nose_chin_angle])

        return features
    except Exception as e:
        print(f"Error saat menghitung fitur geometris: {e}")
        return [np.nan] * 9

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
                    all_features.append([np.nan] * 9)
                    all_labels.append(subfolder_name)

    return all_features, all_labels

# Tentukan folder yang berisi gambar wajah
image_directory = "footage"

# Ekstraksi fitur dari gambar dalam folder
features, labels = process_images_in_directory(image_directory)

# Simpan ke DataFrame
df = pd.DataFrame(features, columns=[
    "Eye Distance", "Nose-Mouth Distance", "Brow Distance", "Nose-Chin Distance",
    "Mouth Corners Distance", "Eye Aspect Ratio", "Mouth Aspect Ratio", "Face Symmetry", "Eye-Nose-Chin Angle"
])
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

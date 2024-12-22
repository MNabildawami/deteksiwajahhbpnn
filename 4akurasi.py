import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Memuat file output aktual dan hasil prediksi
actual_file = 'output/data_uji.csv'
predicted_file = 'output/predictions.csv'

df_actual = pd.read_csv(actual_file)
df_predicted = pd.read_csv(predicted_file)

# Ekstraksi kolom yang relevan
actual_labels = df_actual['Label']
predicted_labels = df_predicted['Predicted_Label']

# Membandingkan label aktual dan prediksi
correct_predictions = (actual_labels == predicted_labels).sum()
total_predictions = len(actual_labels)

# Menghitung akurasi
accuracy = (correct_predictions / total_predictions) * 100
print(f'Tingkat keakuratan metode klasifikasi: {accuracy:.2f}%')

# --- Visualisasi Matriks Kebingungan ---
cm = confusion_matrix(actual_labels, predicted_labels)

# Membuat plot Matriks Kebingungan menggunakan heatmap Seaborn dengan gaya lebih baik
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', linewidths=0.5, linecolor='black',
            xticklabels=actual_labels.unique(), yticklabels=actual_labels.unique(), cbar=True)

# Menambahkan judul dan label
plt.title("Matriks Kebingungan", fontsize=16)
plt.xlabel('Label Prediksi', fontsize=14)
plt.ylabel('Label Aktual', fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# --- Diagram Batang untuk Prediksi Benar vs Salah ---
incorrect_predictions = total_predictions - correct_predictions
labels = ['Benar', 'Salah']
values = [correct_predictions, incorrect_predictions]

# Membuat diagram batang dengan tampilan lebih menarik
plt.figure(figsize=(8, 6))
bars = plt.bar(labels, values, color=['green', 'red'], edgecolor='black')

# Menambahkan garis kisi untuk visibilitas lebih baik
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Menambahkan anotasi di atas batang
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f'{yval}', ha='center', va='bottom', fontsize=12)


# Menambahkan judul dan label pada diagram batang
plt.title('Prediksi Benar vs Salah', fontsize=16)
plt.ylabel('Jumlah Prediksi', fontsize=14)
plt.show()


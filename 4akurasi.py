import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

actual_file = 'output/data_uji_90_10.csv'
predicted_file = 'output/predictions.csv'

df_actual = pd.read_csv(actual_file)
df_predicted = pd.read_csv(predicted_file)

actual_labels = df_actual['Label']
predicted_labels = df_predicted['Predicted_Label']

correct_predictions = (actual_labels == predicted_labels).sum()
total_predictions = len(actual_labels)

accuracy_decimal = correct_predictions / total_predictions
accuracy_percent = accuracy_decimal * 100

print(f"Tingkat keakuratan metode klasifikasi (desimal): {accuracy_decimal:.4f}")
print(f"Tingkat keakuratan metode klasifikasi (persen): {accuracy_percent:.2f}%")

cm = confusion_matrix(actual_labels, predicted_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', linewidths=0.5, linecolor='black',
            xticklabels=actual_labels.unique(), yticklabels=actual_labels.unique(), cbar=True)

plt.title("Matriks Kebingungan", fontsize=16)
plt.xlabel('Label Prediksi', fontsize=14)
plt.ylabel('Label Aktual', fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

incorrect_predictions = total_predictions - correct_predictions
labels = ['Benar', 'Salah']
values = [correct_predictions, incorrect_predictions]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, values, color=['green', 'red'], edgecolor='black')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f'{yval}', ha='center', va='bottom', fontsize=12)

plt.title('Prediksi Benar vs Salah', fontsize=16)
plt.ylabel('Jumlah Prediksi', fontsize=14)
plt.show()

epochs = [1]
accuracies = [accuracy_percent]

plt.figure(figsize=(8, 6))
plt.plot(epochs, accuracies, marker='o', linestyle='-', color='blue', label='Akurasi')

for i, acc in enumerate(accuracies):
    plt.text(epochs[i], acc + 1, f'{acc:.2f}%', ha='center', va='bottom', fontsize=10)

plt.title('Grafik Garis Akurasi', fontsize=16)
plt.xlabel('Percobaan/Epoch', fontsize=14)
plt.ylabel('Persentase Akurasi', fontsize=14)
plt.ylim(0, 100)
plt.xticks(epochs)
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.legend()
plt.show()

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Tentukan folder output
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load dataset
train_file = os.path.join(output_folder, 'data_latih.csv')  # File dataset latih
test_file = os.path.join(output_folder, 'data_uji.csv')  # File dataset uji

df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

# Ekstraksi fitur dan label dari data latih
features = ["Eye Distance", "Nose-Mouth Distance", "Brow Distance", "Nose-Chin Distance", "Mouth Corners Distance"]
x_train = df_train[features].values
y_train = df_train['Label'].values

# Encode label ke dalam bentuk integer
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

# Normalisasi nilai fitur
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(df_test[features].values)

# Definisikan struktur jaringan saraf
n_inputs = len(features)  # Jumlah fitur input
n_hidden = 5  # Jumlah neuron di layer tersembunyi
n_outputs = len(set(y_train))  # Jumlah kelas output
l_rate = 0.3  # Learning rate
n_epoch = 500  # Jumlah epoch

# Inisialisasi jaringan saraf
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = [
        [{'weights': np.random.rand(n_inputs + 1)} for _ in range(n_hidden)],  # Layer input ke layer tersembunyi
        [{'weights': np.random.rand(n_hidden + 1)} for _ in range(n_outputs)]  # Layer tersembunyi ke layer output
    ]
    return network

# Fungsi aktivasi (Sigmoid)
def sigmoid(x, derivative=False):
    if derivative:
        return x * (1.0 - x)
    return 1.0 / (1.0 + np.exp(-x))

# Forward propagation
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = neuron['weights'][-1] + np.dot(neuron['weights'][:-1], inputs)  # Hitung aktivasi
            neuron['output'] = sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Backward propagation untuk menghitung error
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i != len(network) - 1:  # Jika bukan layer output
            for j in range(len(layer)):
                error = sum([neuron['weights'][j] * neuron['delta'] for neuron in network[i + 1]])
                errors.append(error)
        else:  # Jika layer output
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoid(neuron['output'], derivative=True)

# Update bobot jaringan
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1] if i == 0 else [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']

# Melatih jaringan saraf
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        for row in train:
            outputs = forward_propagate(network, row[:-1])
            expected = [0 for _ in range(n_outputs)]
            expected[int(row[-1])] = 1  # Encode one-hot untuk output
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)

# Membuat prediksi
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))  # Pilih indeks dengan nilai output terbesar

# Persiapkan data latih
train_data = np.column_stack((x_train, y_train.astype(int)))  # Tambahkan label ke fitur
network = initialize_network(n_inputs, n_hidden, n_outputs)

# Latih model
train_network(network, train_data, l_rate, n_epoch, n_outputs)

# Prediksi data uji
predictions = [predict(network, row) for row in x_test]
predicted_labels = label_encoder.inverse_transform(predictions)

# Simpan prediksi ke file
df_test['Predicted_Label'] = predicted_labels
predictions_csv_path = os.path.join(output_folder, 'predictions.csv')
df_test.to_csv(predictions_csv_path, index=False)

print(f"Prediksi disimpan di: {predictions_csv_path}")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

# Tentukan folder output
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load dataset
train_file = os.path.join(output_folder, 'data_latih_70_30.csv')
test_file = os.path.join(output_folder, 'data_uji_70_30.csv')

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
n_inputs = len(features)
n_hidden = 10  # Coba dengan 10 neuron tersembunyi
n_outputs = len(set(y_train))
l_rate = 0.05
momentum = 0.9
n_epoch = 1000
patience = 20


# Inisialisasi jaringan saraf
def initialize_network(n_inputs, n_hidden, n_outputs):
    np.random.seed(42)  # Untuk reproduktifitas
    network = [
        [{'weights': np.random.normal(scale=0.01, size=n_inputs + 1), 'velocity': np.zeros(n_inputs + 1)} for _ in
         range(n_hidden)],
        [{'weights': np.random.normal(scale=0.01, size=n_hidden + 1), 'velocity': np.zeros(n_hidden + 1)} for _ in
         range(n_outputs)]
    ]
    return network


# Fungsi aktivasi Leaky ReLU dengan penanganan numerik yang lebih baik
def leaky_relu(x, alpha=0.01, derivative=False):
    if derivative:
        return np.where(x > 0, 1, alpha)
    return np.where(x > 0, x, alpha * x)


# Fungsi aktivasi sigmoid dengan pencegahan overflow
def sigmoid(x, derivative=False):
    # Batasi nilai untuk menghindari overflow eksponensial
    x = np.clip(x, -500, 500)
    if derivative:
        return x * (1.0 - x)
    return 1.0 / (1.0 + np.exp(-x))


# Forward propagation
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = neuron['weights'][-1] + np.dot(neuron['weights'][:-1], inputs)
            neuron['output'] = leaky_relu(activation) if layer != network[-1] else sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Backward propagation untuk menghitung error
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = sum([neuron['weights'][j] * neuron['delta'] for neuron in network[i + 1]])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            derivative = leaky_relu(neuron['output'], derivative=True) if i != len(network) - 1 else sigmoid(
                neuron['output'], derivative=True)
            neuron['delta'] = errors[j] * derivative


# Update bobot jaringan dengan momentum dan regularisasi L2
def update_weights(network, row, l_rate, momentum, l2_lambda=0.01):
    for i in range(len(network)):
        inputs = row[:-1] if i == 0 else [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                gradient = l_rate * neuron['delta'] * inputs[j] + l2_lambda * neuron['weights'][j]
                neuron['velocity'][j] = momentum * neuron['velocity'][j] + gradient
                neuron['weights'][j] += neuron['velocity'][j]
            gradient_bias = l_rate * neuron['delta'] + l2_lambda * neuron['weights'][-1]
            neuron['velocity'][-1] = momentum * neuron['velocity'][-1] + gradient_bias
            neuron['weights'][-1] += neuron['velocity'][-1]


# Learning rate decay
def learning_rate_decay(initial_lr, epoch, decay_rate=0.01):
    return initial_lr / (1 + decay_rate * epoch)


# Melatih jaringan saraf dengan Early Stopping
def train_network(network, train, l_rate, momentum, n_epoch, n_outputs, patience):
    best_loss = float('inf')
    no_improvement = 0
    losses = []
    accuracies = []

    for epoch in range(n_epoch):
        total_loss = 0
        correct_predictions = 0
        for row in train:
            outputs = forward_propagate(network, row[:-1])
            expected = [0 for _ in range(n_outputs)]
            expected[int(row[-1])] = 1
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate, momentum)
            total_loss += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            if np.argmax(outputs) == np.argmax(expected):
                correct_predictions += 1

        accuracy = correct_predictions / len(train)
        losses.append(total_loss)
        accuracies.append(accuracy)

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Update learning rate
        l_rate = learning_rate_decay(l_rate, epoch)

        if total_loss < best_loss:
            best_loss = total_loss
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Visualisasi kerugian dan akurasi pelatihan
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(losses, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Epoch")
    plt.legend()

    plt.show()

    return losses


# Membuat prediksi
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


# Persiapkan data latih
train_data = np.column_stack((x_train, y_train.astype(int)))
network = initialize_network(n_inputs, n_hidden, n_outputs)

# Latih model
losses = train_network(network, train_data, l_rate, momentum, n_epoch, n_outputs, patience)

# Prediksi data uji
predictions = [predict(network, row) for row in x_test]
predicted_labels = label_encoder.inverse_transform(predictions)

# Evaluasi hasil
true_labels = label_encoder.transform(df_test['Label'].values)
accuracy = accuracy_score(true_labels, predictions)
print("Accuracy on test data:", accuracy)
print(classification_report(true_labels, predictions, target_names=label_encoder.classes_))

# Confusion matrix
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Simpan prediksi ke file
df_test['Predicted_Label'] = predicted_labels
predictions_csv_path = os.path.join(output_folder, 'predictions.csv')
df_test.to_csv(predictions_csv_path, index=False)

print(f"Prediksi disimpan di: {predictions_csv_path}")

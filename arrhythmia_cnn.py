# Importing necessary libraries
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import pywt
from sklearn.utils import resample
import keras
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import pickle

# Setting constants
DATA_PATH = '/home/pqb5384/Desktop/mitbih_database/'
WAVELET_NAME = 'sym4'
DECIMATION_THRESHOLD = 0.04
CLASS_LABELS = ['N', 'L', 'R', 'A', 'V']
SAMPLE_COUNT_LIMIT = 10000
SAMPLE_WINDOW = 1080

# Function to reduce noise in the signal
def reduce_noise(input_signal):
    wavelet = pywt.Wavelet(WAVELET_NAME)
    max_level = pywt.dwt_max_level(len(input_signal), wavelet.dec_len)
    coefficients = pywt.wavedec(input_signal, WAVELET_NAME, level=max_level)
    coefficients = [pywt.threshold(i, DECIMATION_THRESHOLD * max(i)) for i in coefficients]
    reconstructed_signal = pywt.waverec(coefficients, WAVELET_NAME)
    return reconstructed_signal

# Gathering and processing the signal data
plot_original_signal = True
plot_denoised_signal = True
data_signals = []
data_labels = []
file_list = os.listdir(DATA_PATH)
csv_files = [file_name for file_name in file_list if file_name.endswith('.csv')]
annotation_files = [file_name for file_name in file_list if file_name.endswith('.txt')]
csv_files.sort()
annotation_files.sort()

for index, file_name in enumerate(csv_files):
    full_path = os.path.join(DATA_PATH, file_name)
    with open(full_path, 'rt') as file:
        reader = csv.reader(file)
        next(reader, None)

        raw_signals = [int(row[1]) for row in reader]

        if plot_original_signal and index == 1:
            plt.figure(figsize=(15, 4))
            plt.title(f'{file_name} Wave')
            plt.plot(raw_signals[0:700])
            plt.show()

        filtered_signals = reduce_noise(raw_signals)

        if plot_denoised_signal and index == 1:
            plt.figure(figsize=(15, 4))
            plt.title(f'{file_name} wave after denoised')
            plt.plot(filtered_signals[0:700])
            plt.show()

        normalized_signals = stats.zscore(filtered_signals)
        if index == 1:
            plt.figure(figsize=(15, 4))
            plt.title(f'{file_name} wave after z-score normalization')
            plt.plot(normalized_signals[0:700])
            plt.show()

        data_signals.append(normalized_signals)

class_counts = [0] * len(CLASS_LABELS)
compiled_data = []
compiled_labels = []

example_beat_printed = False
for index, annotation_file in enumerate(annotation_files):
    with open(os.path.join(DATA_PATH, annotation_file), 'r') as file:
        content = file.readlines()

        for line in content[1:]:
            segments = list(filter(None, line.split(' ')))
            sample_id = int(segments[1])
            label_type = segments[2]

            if label_type in CLASS_LABELS:
                label_index = CLASS_LABELS.index(label_type)

                if class_counts[label_index] < SAMPLE_COUNT_LIMIT and SAMPLE_WINDOW <= sample_id < len(data_signals[index]) - SAMPLE_WINDOW:
                    single_data_point = data_signals[index][sample_id - SAMPLE_WINDOW:sample_id + SAMPLE_WINDOW]
                    compiled_data.append(single_data_point)
                    compiled_labels.append(label_index)
                    class_counts[label_index] += 1

# Preparing datasets for the model
data_frame = pd.DataFrame(compiled_data)
data_frame['label'] = compiled_labels

# Plotting the original class distribution
plt.figure(figsize=(10, 6))
data_frame['label'].value_counts().plot(kind='bar', title='Class Distribution Before Rebalancing')
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.xticks(np.arange(len(CLASS_LABELS)), CLASS_LABELS, rotation=0)
plt.show()

# Preparing datasets for the model
data_frame = pd.DataFrame(compiled_data)
data_frame['label'] = compiled_labels

# Split the data into training and testing before resampling
train_df, test_df = train_test_split(data_frame, test_size=0.2, stratify=data_frame['label'], random_state=42)

# Upsample the training set only
max_count = train_df['label'].value_counts().max()
balanced_data_frames = []
for i in range(len(CLASS_LABELS)):
    current_class = train_df[train_df['label'] == i]
    upsampled_class = resample(current_class, replace=True, n_samples=max_count, random_state=42)
    balanced_data_frames.append(upsampled_class)

balanced_train_df = pd.concat(balanced_data_frames)

# Now we plot the class distribution after resampling only the training data
plt.figure(figsize=(10, 6))
balanced_train_df['label'].value_counts().plot(kind='bar', title='Class Distribution After Rebalancing')
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.xticks(np.arange(len(CLASS_LABELS)), CLASS_LABELS, rotation=0)
plt.show()

# Prepare the labels and samples for training and testing sets
train_labels = to_categorical(balanced_train_df.pop('label'))
test_labels = to_categorical(test_df.pop('label'))
train_samples = np.expand_dims(balanced_train_df.values, axis=2)
test_samples = np.expand_dims(test_df.values, axis=2)

# Creating the model
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(train_samples.shape[1], 1)),
    MaxPooling1D(2),
    Conv1D(128, 3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu', name='rf_train'),
    Dropout(0.5),
    Dense(len(CLASS_LABELS), activation='softmax')
])
model.summary()

# Compiling and training the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_samples, train_labels, validation_data=(test_samples, test_labels), epochs=50, batch_size=32)

# Evaluating the model
performance = model.evaluate(test_samples, test_labels)
print(f"Test Loss: {performance[0]}, Test Accuracy: {performance[1]}")

# Visualizing the results
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

from tensorflow.keras.models import Model
layer_name = 'rf_train'
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

train_features = intermediate_layer_model.predict(train_samples)
test_features = intermediate_layer_model.predict(test_samples)

# Initialize the Random Forest classifier.
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest using the extracted features.
rf_classifier.fit(train_features, np.argmax(train_labels, axis=1))
rf_predictions = rf_classifier.predict(test_features)
rf_predictions_categorical = to_categorical(rf_predictions, num_classes=len(CLASS_LABELS))

rf_test_accuracy = accuracy_score(np.argmax(test_labels, axis=1), rf_predictions)

# Generating predictions from the ensemble model
cnn_predictions = model.predict(test_samples)
ensemble_predictions = 0.5 * (rf_predictions_categorical + cnn_predictions)
ensemble_final_predictions = np.argmax(ensemble_predictions, axis=1)

# Confusion matrix to calculate TP, TN, FP, FN
cm = confusion_matrix(np.argmax(test_labels, axis=1), ensemble_final_predictions)
TP = np.diag(cm)
FP = cm.sum(axis=0) - TP
FN = cm.sum(axis=1) - TP
TN = cm.sum() - (FP + FN + TP)

# Sensitivity, Specificity, Precision, and F1 score calculations
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
precision = precision_score(np.argmax(test_labels, axis=1), ensemble_final_predictions, average='macro')
f1 = f1_score(np.argmax(test_labels, axis=1), ensemble_final_predictions, average='macro')

# Calculating final accuracy
final_accuracy = (TP + TN) / (TP + FP + FN + TN)

# Printing the metrics
print(f"Final Ensemble Model Accuracy: {final_accuracy.mean()}")
print(f"Sensitivity: {sensitivity.mean()}")
print(f"Specificity: {specificity.mean()}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1}")

model.save('/home/pqb5384/Desktop/my_cnn_model.h5')
with open('/home/pqb5384/Desktop/my_rf_model.pk1', 'wb') as file:
  pickle.dump(rf_classifier, file)

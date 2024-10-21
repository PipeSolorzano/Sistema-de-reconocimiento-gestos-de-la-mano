import os
import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, UpSampling2D, LeakyReLU, Dropout, Flatten, Dense, Activation
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from keras.layers import BatchNormalization
from keras.utils import Sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Ruta donde se encuentra la base de datos descargada
data_dir = '/home/pipesolorzano/Documentos/Base_de_datos/Datos_guardados'

# Cargar data_matrix y metadata
data_matrix_path = os.path.join(data_dir, 'data_matrix.npy')
metadata_path = os.path.join(data_dir, 'metadata.npy')

data_matrix = np.load(data_matrix_path)
metadata = np.load(metadata_path, allow_pickle=True)

print(f'La dimensión de data_matrix es: {data_matrix.shape}')

# Generar etiquetas (por simplicidad, asumamos que cada archivo tiene una etiqueta basada en el gesto)
labels = np.array([item['gesture'] for item in metadata])

# Filtrar solo las etiquetas de interés: 13, 15 y 16
valid_classes = [13, 15, 16]
mask = np.isin(labels, valid_classes)
data_matrix = data_matrix[mask]
labels = labels[mask]

# Mapear las clases a índices
class_mapping = {13: 0, 15: 1, 16: 2}
labels = np.array([class_mapping[label] for label in labels])

# Dividir en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(data_matrix, labels, test_size=0.2, random_state=42)

# Redimensionar datos para que se ajusten a (16, 10240, 1) si el modelo espera datos en 4D
X_train = np.expand_dims(X_train, axis=-1)  # Asegurarse de que la forma sea (N, 16, 10240, 1)
X_val = np.expand_dims(X_val, axis=-1)  # Asegurarse de que la forma sea (N, 16, 10240, 1)

print(f'Dimensión x_train: {X_train.shape}')

def build_model(input_shape):
    inputs = Input(shape=input_shape)

    # Módulo del Encoder
    x = Conv2D(8, kernel_size=3, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(16, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dropout(0.5)(x)

    # Módulo del Decoder
    x = UpSampling2D(size=2)(x)
    x = Conv2D(16, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = UpSampling2D(size=2)(x)
    x = Conv2D(8, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dropout(0.5)(x)  # Aplicar un dropout más moderado

    # Aplanar la salida del decoder
    x = Flatten()(x)

    # Añadir capa densa para clasificación
    x = LeakyReLU()(x)
    x = Dense(3)(x)  # Número de clases
    outputs = Activation("softmax")(x)

    model = Model(inputs, outputs)
    return model

# Definir la forma de entrada
input_shape = (16, 10240, 1)

# Construir el modelo
model = build_model(input_shape)
model.summary()

# Configuración del optimizador
initial_lr = 0.001
decay_factor = 0.5
steps_per_epoch = len(X_train) // 2  # Tamaño de lote reducido
decay_steps = 2 * steps_per_epoch  # Aproximadamente cada 2 épocas

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=decay_steps,
    decay_rate=decay_factor,
    staircase=True
)

optimizer = Adam(learning_rate=lr_schedule)

# Compilar el modelo
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Convertir etiquetas a formato one-hot para la pérdida categorical_crossentropy
y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=3)

# Implementar un generador para manejar los datos
class DataGenerator(Sequence):
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.indices = np.arange(len(self.data))
        np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.data[batch_indices]
        batch_labels = self.labels[batch_indices]
        return batch_data, batch_labels

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# Crear generadores
train_generator = DataGenerator(X_train, y_train, batch_size=4)
val_generator = DataGenerator(X_val, y_val, batch_size=4)

# Definir callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(filepath='/home/pipesolorzano/Documentos/Base_de_datos/Datos_guardados/model_bestIV.h5', 
                                    save_best_only=True, monitor='val_loss')

# Entrenar el modelo
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=[early_stopping, model_checkpoint]
)

# Guardar el modelo completo
model_path = '/home/pipesolorzano/Documentos/Base_de_datos/Datos_guardados/model_completeIV.h5'
model.save(model_path)
print(f'Model complete saved to {model_path}')

# Guardar solo los pesos del modelo
weights_path = '/home/pipesolorzano/Documentos/Base_de_datos/Datos_guardados/weights_onlyIV.h5'
model.save_weights(weights_path)
print(f'Model weights saved to {weights_path}')

# Limpiar memoria y poner solo los pesos
tf.keras.backend.clear_session()  # Limpiar la memoria de la sesión actual

# Re-crear el modelo con la misma arquitectura
model = build_model(input_shape)

# Cargar los pesos en el nuevo modelo
model.load_weights(weights_path)
print(f'Model weights loaded from {weights_path}')

# Obtener las predicciones del modelo
y_pred = model.predict(X_val)
y_pred_labels = np.argmax(y_pred, axis=1)  # Convertir predicciones one-hot a etiquetas
y_true_labels = np.argmax(y_val, axis=1)  # Convertir etiquetas one-hot a etiquetas

# Calcular precisión, exhaustividad y F1-score
precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

print(f'Precisión: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Predicción: {y_pred}')

# Reporte de clasificación
report = classification_report(y_true_labels, y_pred_labels)
print('Classification Report:')
print(report)

# Graficar las curvas de entrenamiento
plt.figure(figsize=(12, 6))

# Curva de pérdida
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

# Curva de precisión
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.tight_layout()
plt.savefig('/home/pipesolorzano/Documentos/Base_de_datos/Datos_guardados/curvas_entrenamientoIV.png')
plt.show()


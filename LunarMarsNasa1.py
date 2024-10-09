import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import obspy
from obspy import read
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, precision_recall_curve, f1_score, matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers
import pywt
import warnings
import os
from scipy.fft import fft, ifft
from scipy.signal import stft, detrend
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# ===================== MEJORAS Y MODELOS MATEMÁTICOS ==========================

def navier_stokes_non_newtonian(u, v, p, rho, mu, dt, dx, dy):
    """
    Implementa las ecuaciones de Navier-Stokes para fluidos no newtonianos
    en un espacio bidimensional para el cálculo de la dinámica de fluidos.
    """
    dudx = (u[2:] - u[:-2]) / (2 * dx)
    dudy = (u[:, 2:] - u[:, :-2]) / (2 * dy)
    shear_stress = mu * (dudx + dudy)
    dpdx = (p[2:] - p[:-2]) / (2 * dx)
    dpdy = (p[:, 2:] - p[:, :-2]) / (2 * dy)
    
    u_next = u + dt * (-dpdx + shear_stress / rho)
    v_next = v + dt * (-dpdy + shear_stress / rho)
    
    return u_next, v_next


def thermodynamic_correction(E, u, k, Q, gradient_T, dt):
    """
    Corrección de termodinámica aplicada al análisis de ondas sísmicas.
    """
    term_corr = (np.dot(k, gradient_T) + Q - np.dot(np.gradient(u), E)) * dt
    return term_corr


def ricci_curvature_metric(g, dgdt, R, delta_E, dt):
    """
    Implementa la métrica de curvatura de Ricci para modelar los cambios de 
    topología debido a la dinámica de fluidos en superficies curvas.
    """
    curvature_change = -2 * R * g + delta_E / dt
    new_metric = g + dt * curvature_change
    return new_metric


def centripetal_acceleration(velocity, radius):
    """
    Cálculo de aceleración centrípeta para modelar la fuerza y la velocidad
    en ondas sísmicas en un planeta con geometría esférica.
    """
    acc_centripetal = velocity ** 2 / radius
    return acc_centripetal


# ===================== PROCESAMIENTO DE SEÑALES MULTIPLANETARIAS =====================

def load_and_preprocess_data(file_path, freqmin, freqmax, detrend_signal=False):
    """
    Carga y preprocesa datos sísmicos utilizando ObsPy, maneja datos faltantes y glitches,
    aplica un filtro de paso de banda y normaliza la señal. Ahora soporta diferentes misiones y configuraciones.
    """
    st = read(file_path)
    tr = st[0]
    tr.data = np.nan_to_num(tr.data)
    
    if detrend_signal:
        tr.data = detrend(tr.data)
    
    # Filtro de paso de banda
    tr.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)
    
    # Normalización de datos
    tr.normalize()
    
    return tr


def extract_features_wavelet(tr, scales, waveletname='morl'):
    """
    Extrae coeficientes de la transformada wavelet continua y calcula estadísticas avanzadas.
    """
    coefficients, frequencies = pywt.cwt(tr.data, scales, waveletname, sampling_period=tr.stats.delta)
    
    # Features adicionales (nuevas características)
    mean_coeff = np.mean(coefficients, axis=1)
    std_coeff = np.std(coefficients, axis=1)
    max_coeff = np.max(coefficients, axis=1)
    min_coeff = np.min(coefficients, axis=1)
    energy_coeff = np.sum(np.square(coefficients), axis=1)
    
    skew_coeff = np.mean((coefficients - mean_coeff[:, np.newaxis]) ** 3, axis=1) / std_coeff ** 3
    kurtosis_coeff = np.mean((coefficients - mean_coeff[:, np.newaxis]) ** 4, axis=1) / std_coeff ** 4 - 3
    entropy_coeff = np.apply_along_axis(entropy, 1, coefficients)

    features = np.concatenate([mean_coeff, std_coeff, max_coeff, min_coeff, energy_coeff, skew_coeff, kurtosis_coeff, entropy_coeff])
    
    return features


def extract_stft_features(tr, nperseg=256):
    """
    Extrae características usando la Transformada de Fourier de Ventanas Cortas (STFT).
    """
    _, _, Zxx = stft(tr.data, nperseg=nperseg)
    Zxx_mag = np.abs(Zxx)
    
    mean_stft = np.mean(Zxx_mag, axis=1)
    std_stft = np.std(Zxx_mag, axis=1)
    max_stft = np.max(Zxx_mag, axis=1)
    min_stft = np.min(Zxx_mag, axis=1)
    energy_stft = np.sum(np.square(Zxx_mag), axis=1)
    
    features = np.concatenate([mean_stft, std_stft, max_stft, min_stft, energy_stft])
    
    return features


def assign_label(window_tr, catalog_events, tolerance=2):
    """
    Asigna una etiqueta a una ventana de datos basada en el catálogo de eventos.
    """
    window_start = window_tr.stats.starttime
    window_end = window_tr.stats.endtime
    for event_time in catalog_events:
        if window_start - tolerance <= event_time <= window_end + tolerance:
            return 1  # Evento detectado en la ventana
    return 0  # No hay evento en la ventana


def process_window(tr, i, window_size, step_size, scales, catalog_events):
    """
    Procesa una ventana de datos, extrayendo características y asignando etiquetas.
    """
    start = i * step_size
    end = start + window_size
    window_data = tr.data[start:end]
    window_tr = tr.copy()
    window_tr.data = window_data
    window_tr.stats.starttime = tr.stats.starttime + start * tr.stats.delta
    window_tr.stats.endtime = tr.stats.starttime + end * tr.stats.delta

    # Extraer características
    wavelet_features = extract_features_wavelet(window_tr, scales)
    stft_features = extract_stft_features(window_tr)

    features = np.concatenate([wavelet_features, stft_features])

    # Asignar etiqueta
    label = assign_label(window_tr, catalog_events)

    return features, label


def parallel_create_dataset(tr, window_size, step_size, scales, catalog_events, augment=False, n_jobs=-1):
    """
    Crea ventanas de datos y extrae características y etiquetas para cada ventana de manera paralelizada.
    """
    num_samples = len(tr.data)
    num_steps = (num_samples - window_size) // step_size + 1
    results = Parallel(n_jobs=n_jobs)(delayed(process_window)(tr, i, window_size, step_size, scales, catalog_events) for i in range(num_steps))
    
    data, labels = zip(*results)
    data = np.array(data)
    labels = np.array(labels)

    if augment:
        data = augment_data(data)

    return data, labels


def augment_data(data):
    """
    Implementa técnicas de aumento de datos: añade ruido gaussiano leve y aplica desplazamiento temporal.
    """
    augmented_data = data.copy()
    # Añadir ruido gaussiano
    noise_factor = 0.005
    noise = noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    augmented_data += noise

    # Aplicar desplazamiento temporal
    shift_factor = 0.1
    shift_samples = int(shift_factor * data.shape[1])
    augmented_data = np.roll(augmented_data, shift_samples, axis=1)
    
    return augmented_data


# ===================== AMPLIACIÓN DEL MODELO DE RED NEURONAL =========================

def build_deep_model(input_shape):
    """
    Construye un modelo de red neuronal profunda para la detección de eventos sísmicos.
    Añade Batch Normalization y dropout para mejorar la generalización.
    """
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Reshape((input_shape[0], 1)))

    # Capas convolucionales mejoradas con Batch Normalization
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Conv1D(256, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))

    # Capas LSTM para captar las relaciones temporales
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.LSTM(64))

    # Aplanar y capas densas
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # Regularización con dropout
    model.add(layers.Dense(1, activation='sigmoid'))

    # Optimizador AdamW con regularización
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# ===================== DETECCIÓN DE EVENTOS =========================

def detect_events_in_signal(tr, model, window_size, step_size, scales):
    """
    Aplica el modelo entrenado para detectar eventos en nuevos datos sísmicos.
    """
    events = []
    times = []
    num_samples = len(tr.data)
    num_steps = (num_samples - window_size) // step_size + 1
    for i in range(num_steps):
        start = i * step_size
        end = start + window_size
        if end > num_samples:
            break
        window_data = tr.data[start:end]
        window_tr = tr.copy()
        window_tr.data = window_data
        window_tr.stats.starttime = tr.stats.starttime + start * tr.stats.delta

        wavelet_features = extract_features_wavelet(window_tr, scales)
        stft_features = extract_stft_features(window_tr)

        features = np.concatenate([wavelet_features, stft_features])
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        if prediction > 0.5:
            events.append(window_data)
            times.append(window_tr.stats.starttime)

    return events, times


# ===================== EVALUACIÓN Y VISUALIZACIÓN ===================

def save_detections_to_csv(events, times, output_file='detections.csv'):
    """
    Guarda los tiempos y eventos detectados en un archivo CSV.
    """
    detections = pd.DataFrame({
        'time_abs': [t.strftime('%Y-%m-%dT%H:%M:%S.%f') for t in times],
        'event_data': [str(event) for event in events]
    })
    detections.to_csv(output_file, index=False)
    print(f"Detections saved to {output_file}")


def plot_roc_curve(y_val, y_pred_val):
    """
    Grafica la curva ROC y calcula el AUC.
    """
    fpr, tpr, _ = roc_curve(y_val, y_pred_val)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_precision_recall_curve(y_val, y_pred_val):
    """
    Grafica la curva Precision-Recall.
    """
    precision, recall, _ = precision_recall_curve(y_val, y_pred_val)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()


def plot_training_history(history):
    """
    Grafica la historia del entrenamiento (loss y accuracy).
    """
    plt.figure(figsize=(12, 6))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()


# ===================== FLUJO DE TRABAJO PRINCIPAL ====================

def main():
    planet_datasets = {
        "marte": {'train': 'ModeloMatematico\conjuntos\Marte\output.sac', 'test': 'marte_test.sac', 'catalog': 'marte_catalog.csv'},
        "luna": {'train': 'luna_train.sac', 'test': 'luna_test.sac', 'catalog': 'luna_catalog.csv'}
    }
    
    # Parámetros generales
    freqmin = 0.1   # Frecuencia mínima para el filtrado
    freqmax = 5.0   # Frecuencia máxima para el filtrado
    window_size = 2048  # Tamaño de la ventana
    step_size = 512     # Tamaño del paso
    scales = np.arange(1, 128)

    for planet, files in planet_datasets.items():
        print(f"\n\nProcesando datos para {planet.capitalize()}...\n")
        file_path = files['train']
        test_file_path = files['test']
        catalog_path = files['catalog']

        # Cargar y preprocesar datos de entrenamiento
        tr = load_and_preprocess_data(file_path, freqmin, freqmax)

        # Cargar catálogo de eventos
        catalog = pd.read_csv(catalog_path)
        catalog_events = [obspy.UTCDateTime(t) for t in catalog['event_time']]

        # Crear conjunto de datos de entrenamiento
        data, labels = parallel_create_dataset(tr, window_size, step_size, scales, catalog_events, augment=True)

        # Validación cruzada k-fold estratificada
        skf = StratifiedKFold(n_splits=5)
        for train_index, val_index in skf.split(data, labels):
            X_train, X_val = data[train_index], data[val_index]
            y_train, y_val = labels[train_index], labels[val_index]

            # Oversampling con SMOTE
            sm = SMOTE()
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

            # Manejo del desbalanceo de clases
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train_res), y=y_train_res)
            class_weights = {i: class_weights[i] for i in range(len(class_weights))}

            # Construir y entrenar el modelo
            input_shape = X_train_res.shape[1:]
            model = build_deep_model(input_shape)
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            checkpoint = ModelCheckpoint(f'best_model_{planet}.h5', save_best_only=True)
            
            history = model.fit(X_train_res, y_train_res, epochs=20, batch_size=64, validation_data=(X_val, y_val), 
                                class_weight=class_weights, callbacks=[early_stopping, checkpoint])

            # Evaluación en el conjunto de validación
            y_pred_val = model.predict(X_val).ravel()
            y_pred_binary = (y_pred_val > 0.5).astype("int32")
            print(f"Resultados en el conjunto de validación para {planet.capitalize()}:")
            print(classification_report(y_val, y_pred_binary))

            # Métricas adicionales
            f1 = f1_score(y_val, y_pred_binary)
            mcc = matthews_corrcoef(y_val, y_pred_binary)
            print(f"F1-Score: {f1:.4f}, MCC: {mcc:.4f}")

            # Matriz de confusión
            print(f"Matriz de confusión en el conjunto de validación para {planet.capitalize()}:")
            print(confusion_matrix(y_val, y_pred_binary))

            # Curvas ROC y Precision-Recall
            plot_roc_curve(y_val, y_pred_val)
            plot_precision_recall_curve(y_val, y_pred_val)

            # Graficar la historia del entrenamiento
            plot_training_history(history)

        # Cargar y preprocesar datos de prueba
        tr_test = load_and_preprocess_data(test_file_path, freqmin, freqmax)

        # Detectar eventos en datos de prueba
        events, times = detect_events_in_signal(tr_test, model, window_size, step_size, scales)

        # Verificar si se detectaron eventos
        if not events:
            print(f"No se detectaron eventos en los datos de prueba para {planet.capitalize()}.")
        else:
            save_detections_to_csv(events, times, output_file=f'detections_{planet}.csv')


if __name__ == "__main__":
    main()

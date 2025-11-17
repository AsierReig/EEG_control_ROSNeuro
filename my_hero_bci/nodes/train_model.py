#!/usr/bin/env python3
import rosbag
import numpy as np
from scipy.signal import welch, butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib
import argparse

# ---------------------------------------------------
# Filtrado band-pass
# ---------------------------------------------------
def bandpass_filter(epoch, lowcut=8, highcut=30, fs=256, order=4):
    """Filtra epoch en la banda [lowcut, highcut] Hz"""
    b, a = butter(order, [lowcut/(0.5*fs), highcut/(0.5*fs)], btype='band')
    return filtfilt(b, a, epoch, axis=1)

# ---------------------------------------------------
# Extracción de características
# ---------------------------------------------------
def extract_bandpower(epoch, fs=256, bands=[(8,12),(13,30)]):
    """Calcula log-bandpower por canal y banda"""
    feats = []
    for ch in epoch:
        f, Pxx = welch(ch, fs=fs, nperseg=fs)
        for lo, hi in bands:
            idx = (f >= lo) & (f <= hi)
            feats.append(np.log(np.trapz(Pxx[idx], f[idx]) + 1e-12))
    return np.array(feats)

# ---------------------------------------------------
# Carga de datos
# ---------------------------------------------------
def load_bag_data(bag_path, eeg_topic='/neurodata', event_topic='/neuroevent'):
    """Extrae datos EEG y eventos desde rosbag"""
    bag = rosbag.Bag(bag_path)
    eeg_data = []
    event_list = []
    channel_labels = []

    # Leer EEG
    for _, msg, _ in bag.read_messages(topics=[eeg_topic]):
        eeg = np.array(msg.eeg.data).reshape(msg.eeg.info.nchannels, msg.eeg.info.nsamples)
        eeg_data.append((msg.header.stamp.to_sec(), eeg))
        channel_labels = msg.eeg.info.labels

    # Leer eventos
    for _, msg, _ in bag.read_messages(topics=[event_topic]):
        event_list.append((msg.header.stamp.to_sec(), msg.event))

    bag.close()
    print(f"Loaded {len(eeg_data)} EEG epochs and {len(event_list)} events from {bag_path}")
    return eeg_data, event_list, channel_labels

# ---------------------------------------------------
# Selección de canales
# ---------------------------------------------------
def select_channels(eeg_data, ch_labels, wanted=['eeg:4','eeg:5','eeg:6']):
    idx = [i for i, ch in enumerate(ch_labels) if ch in wanted]
    print(f"Selected channels: {[ch_labels[i] for i in idx]}")
    filtered_data = []
    for t, epoch in eeg_data:
        filtered_data.append((t, epoch[idx, :]))
    return filtered_data

# ---------------------------------------------------
# Segmentación por eventos
# ---------------------------------------------------
def segment_epochs_continuous(eeg_data, events, fs=256, trial_len=6.0, win=1.0, overlap=0.5):
    """
    Corta la señal continua según los eventos y genera ventanas para entrenamiento
    trial_len: duración de cada trial en segundos
    win: ventana para características
    overlap: solapamiento entre ventanas
    """
    X, y = [], []

    # Concatenar todo EEG en una sola matriz continua
    eeg_full = np.hstack([epoch for _, epoch in eeg_data])  # nchannels x total_samples
    time_full = np.hstack([t + np.arange(epoch.shape[1])/fs for t, epoch in eeg_data])  # tiempo de cada muestra

    samples_win = int(fs * win)
    step = int(samples_win * (1 - overlap))
    samples_trial = int(fs * trial_len)

    for ev_time, ev_code in events:
        # índice de inicio del trial
        idx_start = np.argmin(np.abs(time_full - ev_time))
        idx_end = idx_start + samples_trial
        if idx_end > eeg_full.shape[1]:
            # No hay suficientes muestras hasta el final
            continue

        trial = eeg_full[:, idx_start:idx_end]
        # Filtrado banda mu/beta
        trial = bandpass_filter(trial, fs=fs)

        # Ventanas dentro del trial
        for start in range(0, trial.shape[1] - samples_win + 1, step):
            window = trial[:, start:start+samples_win]
            X.append(window)
            y.append(ev_code)

    return np.array(X), np.array(y)

# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="/home/tartaria/catkin_ws/src/my_hero_bci/nodes/2025-11-12-09-57-13.bag")
    parser.add_argument("--model", default="model1.joblib", help="Nombre del modelo de salida")
    args = parser.parse_args()

    eeg_data, events, ch_labels = load_bag_data(args.bag)
    eeg_data = select_channels(eeg_data, ch_labels)

    X, y = segment_epochs_continuous(eeg_data, events)
    print(f"Segmented into {len(X)} windows")

    X_feats = np.array([extract_bandpower(x) for x in X])
    print(f"Feature shape: {X_feats.shape}")

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_feats, y)
    joblib.dump(lda, args.model)
    print(f"Model saved to {args.model}")

if __name__ == "__main__":
    main()




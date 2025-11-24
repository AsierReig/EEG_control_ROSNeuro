#!/usr/bin/env python3
"""
Entrena un LDA para 3 clases (0=rest,1=open,2=close) desde un rosbag con /neurodata (NeuroFrame)
y /neuroevent (NeuroEvent). Extrae ventanas peri-evento, filtra y calcula bandpower (mu/beta).
"""

import rosbag
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
import argparse
import sys

# ----- Params de preprocesado -----
WANTED_CHANNELS_DEFAULT = ['eeg:4','eeg:5','eeg:6']  # cambiar si en tu bag son otros
LOWCUT = 8.0
HIGHCUT = 30.0
FILTER_ORDER = 4

def bandpass_filter(epoch, lowcut=LOWCUT, highcut=HIGHCUT, fs=256, order=FILTER_ORDER):
    b, a = butter(order, [lowcut/(0.5*fs), highcut/(0.5*fs)], btype='band')
    # epoch: (nchan, nsamples)
    return filtfilt(b, a, epoch, axis=1)

def extract_bandpower(epoch, fs=256, bands=[(8,12),(13,30)]):
    feats = []
    for ch in range(epoch.shape[0]):
        f, Pxx = welch(epoch[ch, :], fs=fs, nperseg=min(256, epoch.shape[1]))
        for lo, hi in bands:
            idx = (f >= lo) & (f <= hi)
            bp = np.trapz(Pxx[idx], f[idx]) if np.any(idx) else 1e-12
            feats.append(np.log(bp + 1e-12))
    return np.array(feats, dtype=np.float32)

def load_bag(bag_path, eeg_topic='/neurodata', event_topic='/neuroevent'):
    bag = rosbag.Bag(bag_path)
    eeg_data = []   # list of (t_start_sec, ndarray nch x nsamples)
    events = []     # list of (t_event_sec, event_code)
    ch_labels = None
    fs = None

    # Leer EEG messages (NeuroFrame)
    for _, msg, _ in bag.read_messages(topics=[eeg_topic]):
        # sample rate is msg.sr (as observed)
        if fs is None:
            try:
                fs = int(msg.sr)
            except:
                fs = int(256)
        ns = msg.eeg.info.nsamples
        nc = msg.eeg.info.nchannels
        data = np.array(msg.eeg.data, dtype=np.float32)
        if data.size != ns * nc:
            # intenta reorganizar si el orden fuera distinto
            # pero habitualmente es nc * ns
            pass
        # reshape a (ns, nc) y luego transpose a (nc, ns)
        try:
            arr = data.reshape((ns, nc)).T  # (nc, ns)
        except Exception as e:
            # último recurso: intentar la otra forma
            arr = data.reshape((nc, ns))
        eeg_data.append((msg.header.stamp.to_sec(), arr))
        ch_labels = [str(x) for x in msg.eeg.info.labels]
    # Leer eventos NeuroEvent
    for _, msg, _ in bag.read_messages(topics=[event_topic]):
        # msg.event es un int: 0,1,2
        events.append((msg.header.stamp.to_sec(), int(msg.event)))
    bag.close()
    if fs is None:
        fs = 256
    print(f"Loaded {len(eeg_data)} EEG chunks and {len(events)} events. fs={fs}, channels: {ch_labels}")
    return eeg_data, events, fs, ch_labels

def concat_eeg(eeg_data):
    """Concatena la señal continua y genera vector de tiempos por muestra (segundos)."""
    # eeg_data list of (t_chunk_start, arr(nc, ns))
    if len(eeg_data) == 0:
        return None, None
    fs_guess = None
    # construir arrays
    signals = [arr for _, arr in eeg_data]
    times = []
    for t0, arr in eeg_data:
        ns = arr.shape[1]
        times.append(t0 + np.arange(ns) / float(fs_guess if fs_guess else 256.0))
        if fs_guess is None:
            # inferir fs as 1 / delta between samples if possible (no robust pero sirve)
            fs_guess = 256.0
    eeg_full = np.hstack(signals)  # (nchan, total_samples)
    # recompute times properly using fs=256 or your fs (we will pass fs externally)
    return eeg_full

def segment_peri_event(eeg_full, times_full, events, fs, pre=2.0, dur=6.0, post=2.0, win=1.0, overlap=0.5):
    """
    eeg_full: (nchan, total_samples)
    times_full: array of timestamps for each sample (len total_samples)
    events: list of (t_event, label)
    devuelve X windows y y labels
    """
    X, y = [], []
    samples_pre = int(pre * fs)
    samples_dur = int(dur * fs)
    samples_post = int(post * fs)
    win_s = int(win * fs)
    step = int(win_s * (1 - overlap))

    # build times array if not provided
    # times_full must be length total_samples
    for ev_time, ev_code in events:
        # encontrar índice en times_full más cercano a ev_time
        idx = np.argmin(np.abs(times_full - ev_time))
        start = idx - samples_pre
        end = idx + samples_dur + samples_post
        if start < 0 or end > eeg_full.shape[1]:
            continue
        trial = eeg_full[:, start:end]  # (nchan, samples_trial)
        # filtrar
        trial = bandpass_filter(trial, fs=fs)
        # sliding windows
        for s in range(0, trial.shape[1] - win_s + 1, step):
            window = trial[:, s:s+win_s]
            X.append(window)
            y.append(ev_code)
    if len(X) == 0:
        return np.zeros((0,)), np.zeros((0,))
    return np.array(X), np.array(y)

def select_channel_indices(ch_labels, wanted):
    # Intenta detectar por substring si no coincide exactamente
    idx = []
    for w in wanted:
        found = False
        for i, ch in enumerate(ch_labels):
            if w.lower() in ch.lower():
                idx.append(i)
                found = True
                break
        if not found:
            # buscar por exact match
            for i, ch in enumerate(ch_labels):
                if ch == w:
                    idx.append(i)
                    found = True
                    break
    return sorted(list(set(idx)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="ruta al .bag")
    parser.add_argument("--model", default="model1.joblib", help="modelo salida")
    parser.add_argument("--wanted", nargs='+', default=WANTED_CHANNELS_DEFAULT, help="canales deseados")
    parser.add_argument("--pre", type=float, default=2.0)
    parser.add_argument("--dur", type=float, default=6.0)
    parser.add_argument("--post", type=float, default=2.0)
    parser.add_argument("--win", type=float, default=1.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    args = parser.parse_args()

    eeg_chunks, events, fs, ch_labels = load_bag(args.bag)
    if len(eeg_chunks) == 0:
        print("No hay datos EEG en el bag.")
        sys.exit(1)
    if len(events) == 0:
        print("No hay eventos en el bag.")
        sys.exit(1)

    # concatenar señal continua y generar vector de tiempos
    # construir eeg_full and times_full properly using fs
    signals = [arr for _, arr in eeg_chunks]
    eeg_full = np.hstack(signals)  # (nchan, total_samples)
    # construir times_full as continuous increasing using first chunk time and fs
    start_time = eeg_chunks[0][0]
    total_samples = eeg_full.shape[1]
    times_full = start_time + np.arange(total_samples) / float(fs)

    # seleccionar índices de canales
    idxs = select_channel_indices(ch_labels, args.wanted)
    if not idxs:
        print("No se encontraron canales solicitados. Canales disponibles:", ch_labels)
        sys.exit(1)
    eeg_full = eeg_full[idxs, :]
    selected_labels = [ch_labels[i] for i in idxs]
    print("Selected channels:", selected_labels)

    # segmentación peri-evento
    X, y = segment_peri_event(eeg_full, times_full, events, fs,
                              pre=args.pre, dur=args.dur, post=args.post,
                              win=args.win, overlap=args.overlap)

    print(f"Segmented into {len(X)} windows. Labels distribution:", np.unique(y, return_counts=True))
    if len(X) == 0:
        print("No se generaron ventanas. Revisa duración del trial / fs / alignment.")
        sys.exit(1)

    # extraer features
    X_feats = np.array([extract_bandpower(window, fs=fs) for window in X])
    print("Feature shape:", X_feats.shape)

    # entrenar LDA
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_feats, y)
    joblib.dump({'model': clf, 'channels': selected_labels, 'fs': fs}, args.model)
    print("Modelo guardado en", args.model)


    # === Métricas ===
    loaded = joblib.load(args.model)
    clf = loaded['model']  # EXTRAER EL LDA REAL
    y_pred = clf.predict(X_feats)
    acc = accuracy_score(y, y_pred)
    print("Accuracy:", acc)

    print("\n=== Classification report ===")
    print(classification_report(y, y_pred))

    print("\n=== Confusion matrix ===")
    print(confusion_matrix(y, y_pred))

    # === Matriz de confusión gráfica ===
    plt.imshow(confusion_matrix(y, y_pred), cmap='Blues')
    plt.colorbar()
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

if __name__ == "__main__":
    main()





#!/usr/bin/env python3
import rosbag
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

BANDS = {"mu": (8, 12), "beta": (13, 30)}

def butter_bandpass(lo, hi, fs, order=4):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band')
    return b, a

def compute_bandpower(signal, fs, lo, hi):
    f, Pxx = welch(signal, fs=fs, nperseg=256)
    idx = (f >= lo) & (f <= hi)
    return np.trapz(Pxx[idx], f[idx])

def main(bagfile):
    print("Cargando rosbag:", bagfile)
    bag = rosbag.Bag(bagfile)

    events = []
    eeg_segments = []
    sr = None

    # ---- Primera pasada: recolectar EEG y eventos ----
    for topic, msg, t in bag.read_messages():
        if topic == "/neurodata":

            # SR correcto
            if sr is None:
                sr = msg.sr
            elif sr != msg.sr:
                print("[WARN] Frecuencia muestreo cambiada dentro del bag")

            nsamp = msg.eeg.info.nsamples
            nch = msg.eeg.info.nchannels

            arr = np.array(msg.eeg.data).reshape(nsamp, nch)
            eeg_segments.append(arr)

        elif topic == "/neuroevent":
            events.append((t.to_sec(), msg.event))

    bag.close()

    if len(eeg_segments) == 0:
        print("ERROR: No se encontraron datos en /neurodata")
        return

    eeg_data = np.vstack(eeg_segments)  # (samples, channels)
    print("EEG shape:", eeg_data.shape)
    print("Eventos:", len(events))
    print("SR detectada:", sr)

    # ---- Parámetros de ventana ----
    win_pre = 1.0   # 1 s antes del evento
    win_post = 3.0  # 3 s después
    samples_pre = int(win_pre * sr)
    samples_post = int(win_post * sr)

    # Estructura donde guardaremos los trials por clase
    class_trials = {0: [], 1: [], 2: []}

    # Punto inicial de referencia temporal
    t0 = events[0][0]

    # ---- Extraer trials por evento ----
    for t_event, code in events:
        # Convertir tiempo a índice
        idx = int((t_event - t0) * sr)

        # Asegurar que no se sale del array
        if idx - samples_pre < 0 or idx + samples_post >= len(eeg_data):
            continue

        segment = eeg_data[idx - samples_pre : idx + samples_post, :]

        if code in class_trials:
            class_trials[code].append(segment)

    # ---- ERD / ERS por clase ----
    print("\n=== Bandpower μ y β por clase ===")
    for cls in [0, 1, 2]:
        trials = class_trials[cls]
        if len(trials) == 0:
            print(f"Clase {cls}: sin datos")
            continue

        avg_mu = []
        avg_beta = []
        for tr in trials:
            # usa canal 3 (eeg:3) por defecto; cámbialo si quieres C3/C4
            ch = tr[:, 3]

            mu = compute_bandpower(ch, sr, *BANDS["mu"])
            beta = compute_bandpower(ch, sr, *BANDS["beta"])

            avg_mu.append(mu)
            avg_beta.append(beta)

        print(f"\nClase {cls} ({len(trials)} trials):")
        print("  Mu:", np.mean(avg_mu))
        print("  Beta:", np.mean(avg_beta))

    # ---- Gráfico comparativo ERD/ERS ----
    labels = ["Rest", "Open", "Close"]
    mu_vals = []
    beta_vals = []

    for cls in [0, 1, 2]:
        trials = class_trials[cls]
        if len(trials) > 0:
            seg = np.mean(trials, axis=0)
            ch = seg[:, 3]  # canal 3
            mu_vals.append(compute_bandpower(ch, sr, 8, 12))
            beta_vals.append(compute_bandpower(ch, sr, 13, 30))
        else:
            mu_vals.append(0)
            beta_vals.append(0)

    plt.figure()
    x = np.arange(3)
    plt.bar(x-0.15, mu_vals, width=0.3, label="Mu")
    plt.bar(x+0.15, beta_vals, width=0.3, label="Beta")
    plt.xticks(x, labels)
    plt.title("ERD / ERS por clase")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=str, required=True)
    args = parser.parse_args()
    main(args.bag)


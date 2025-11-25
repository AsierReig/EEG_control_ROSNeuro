#!/usr/bin/env python3
"""
1) Reconstruye señal EEG CONTINUA desde un rosbag con mensajes /neurodata (NeuroFrame).
2) Alinea eventos de /neuroevent (NeuroEvent).
3) Extrae epochs peri-evento robustos (usa relleno/interpolación si faltan muestras).
4) Calcula TFR por epoch y produce ERD/ERS (%) respecto a baseline (-1..0s).
5) Plotea heatmaps TFR promedio por clase y resumen de bandpower.

Notas:
- Ajusta channel_index (por defecto 3) o use --chan-index.
- Este script intenta reproducir la reconstrucción tipo ring-buffer que usa ROS-Neuro.
"""

import rosbag
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import argparse
import math
from tqdm import tqdm

# ---------------- params por defecto ----------------
DEFAULT_PRE = 1.0    # segundos antes del evento
DEFAULT_POST = 3.0   # segundos despues del evento
DEFAULT_NPERSEG = 128
DEFAULT_CH_INDEX = 3  # canal para mostrar bandpower (ajusta si necesitas C3/C4/Cz)

# ---------------- utilidades ----------------
def safe_reshape_data(data_flat, ns, nc):
    """
    Intenta reorganizar data_flat en (nc, ns).
    msg.eeg.data puede venir como ns*nch row-major (ns x nch) o nch x ns.
    """
    arr = np.array(data_flat, dtype=np.float32)
    if arr.size == ns * nc:
        try:
            # hemos observado que normalmente viene (ns, nc)
            return arr.reshape((ns, nc)).T  # -> (nc, ns)
        except:
            return arr.reshape((nc, ns))
    else:
        # tamaño inesperado, devolver zeros y avisar
        print("WARNING: tamaño de data inesperado: got", arr.size, "expected", ns*nc)
        return np.zeros((nc, ns), dtype=np.float32)

def interpolate_gaps(eeg_full):
    """
    eeg_full: (nchan, total_samples) con NaNs en huecos.
    Rellena NaNs por interpolación lineal en eje temporal por canal.
    """
    nchan, nS = eeg_full.shape
    for ch in range(nchan):
        v = eeg_full[ch, :]
        nan_idx = np.isnan(v)
        if np.all(nan_idx):
            # todo NaN -> poner 0
            eeg_full[ch, :] = 0.0
            continue
        if np.any(nan_idx):
            good = ~nan_idx
            xp = np.flatnonzero(good)
            fp = v[good]
            xi = np.flatnonzero(nan_idx)
            v[nan_idx] = np.interp(xi, xp, fp)
            eeg_full[ch, :] = v
    return eeg_full

def compute_tfr(epoch, sr, nperseg=DEFAULT_NPERSEG):
    """
    epoch: (nchan, ns_epoch)
    devuelve f, t, tfr: tfr shape (nchan, nfreq, nt)
    """
    nch = epoch.shape[0]
    tfr_list = []
    f_list = None
    t_list = None
    for ch in range(nch):
        f, t, Z = stft(epoch[ch, :], fs=sr, nperseg=nperseg)
        if f_list is None:
            f_list = f
            t_list = t
        tfr_list.append(np.abs(Z)**2)  # potencia
    return f_list, t_list, np.array(tfr_list)  # (nchan, nfreq, nt)

def erd_ers_percent(tfr, t, baseline_time_window=(-1.0, 0.0)):
    """
    tfr: (nfreq, nt) OR (nchan, nfreq, nt) if passed averaged per channel
    t: times relative to epoch start (in seconds)
    baseline_time_window: tuple (t0,t1) relative to epoch start
    devuelve ERD% array de la misma forma que tfr (por freq x time)
    """
    # t puede venir desde 0..T where we want baseline in [-1..0]. We'll assume t is relative to epoch start and that
    # epoch start corresponds to tmin (e.g. -pre). In our workflow we'll shift t later so 0==event.
    bs_idx = np.where((t >= baseline_time_window[0]) & (t <= baseline_time_window[1]))[0]
    if bs_idx.size == 0:
        return None
    baseline = np.mean(tfr[..., bs_idx], axis=-1, keepdims=True)  # mean across baseline times
    # avoid zero baseline
    baseline[baseline == 0] = 1e-12
    erd = (tfr - baseline) / baseline * 100.0
    return erd

# ---------------- main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bag", required=True, help="ruta al rosbag")
    p.add_argument("--pre", type=float, default=DEFAULT_PRE)
    p.add_argument("--post", type=float, default=DEFAULT_POST)
    p.add_argument("--chan-index", type=int, default=DEFAULT_CH_INDEX,
                   help="índice (0-based) del canal para gráficos de bandpower (ej Cz)")
    p.add_argument("--nperseg", type=int, default=DEFAULT_NPERSEG)
    args = p.parse_args()

    bagfile = args.bag
    print("Cargando bag:", bagfile)
    bag = rosbag.Bag(bagfile)

    # Leemos todos los paquetes de /neurodata y /neuroevent, manteniendo timestamps
    neuro_msgs = []   # list of (t_msg_sec, ns, nc, arr(nc, ns), sr, labels)
    events = []       # list of (t_event_sec, event_code, desc)
    for topic, msg, t in bag.read_messages(topics=["/neurodata", "/neuroevent"]):
        if topic == "/neurodata":
            # extraer sr, ns, nc, labels
            try:
                sr = int(msg.sr)
            except:
                # fallback
                sr = int(msg.eeg.info.sampling_rate) if hasattr(msg.eeg.info, 'sampling_rate') else 256
            ns = int(msg.eeg.info.nsamples)
            nc = int(msg.eeg.info.nchannels)
            labels = [str(x) for x in msg.eeg.info.labels]
            # reshape safe
            arr = safe_reshape_data(msg.eeg.data, ns, nc)
            neuro_msgs.append((msg.header.stamp.to_sec(), ns, nc, arr, sr, labels))
        elif topic == "/neuroevent":
            try:
                ev_code = int(msg.event)
            except:
                ev_code = None
            events.append((msg.header.stamp.to_sec(), ev_code, getattr(msg, "description", "")))
    bag.close()
    print(f"Paquetes EEG leídos: {len(neuro_msgs)} | eventos leídos: {len(events)}")
    if len(neuro_msgs) == 0:
        print("No hay /neurodata en el bag. Abortando.")
        return
    if len(events) == 0:
        print("No hay /neuroevent en el bag. Abortando.")
        return

    # Ordenar por timestamp por si acaso
    neuro_msgs.sort(key=lambda x: x[0])
    events.sort(key=lambda x: x[0])

    # inferir sr (tomamos el sr del primer paquete)
    sr = neuro_msgs[0][4]
    print("Frecuencia SR inferida:", sr)

    # construir timeline continuo: start_time -> end_time
    t_start = neuro_msgs[0][0]
    last_msg_time, last_ns, last_nc, last_arr, last_sr, _ = neuro_msgs[-1]
    # last sample time = last_msg_time + (ns-1)/sr
    t_end = last_msg_time + (last_ns - 1) / float(sr)
    total_duration = t_end - t_start
    total_samples = int(math.ceil(total_duration * sr)) + 1
    print(f"Timeline: {t_start} .. {t_end}  duration {total_duration:.3f}s  samples={total_samples}")

    # choose nch from first packet (we assume fixed channels across messages)
    nch = neuro_msgs[0][2]

    # allocate with NaNs
    eeg_full = np.full((nch, total_samples), np.nan, dtype=np.float32)

    # fill eeg_full: for each packet compute start index
    for (t_msg, ns, nc, arr, sr_msg, labels) in neuro_msgs:
        # compute start sample index relative to t_start
        start_idx = int(round((t_msg - t_start) * sr))
        end_idx = start_idx + ns
        if start_idx < 0:
            # clip start
            clip = -start_idx
            if clip >= ns:
                continue
            arr = arr[:, clip:]
            start_idx = 0
            ns = arr.shape[1]
            end_idx = start_idx + ns
        if end_idx > total_samples:
            # clip to fit
            ns_allowed = max(0, total_samples - start_idx)
            arr = arr[:, :ns_allowed]
            end_idx = start_idx + ns_allowed
        if arr.shape[0] != nch:
            # if channel count differs, try to adapt
            if arr.shape[0] < nch:
                # pad channels with zeros
                pad = np.zeros((nch - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
                arr = np.vstack([arr, pad])
            else:
                arr = arr[:nch, :]
        eeg_full[:, start_idx:end_idx] = arr

    # If some samples still NaN, interpolate per channel
    nan_count_before = np.sum(np.isnan(eeg_full))
    print("NaNs before interp:", nan_count_before)
    eeg_full = interpolate_gaps(eeg_full)
    nan_count_after = np.sum(np.isnan(eeg_full))
    print("NaNs after interp:", nan_count_after)

    # Build global times array
    times_full = t_start + np.arange(eeg_full.shape[1]) / float(sr)

    # Now extract epochs per event using continuous eeg_full
    pre = args.pre
    post = args.post
    samples_pre = int(round(pre * sr))
    samples_post = int(round(post * sr))
    epoch_len = samples_pre + samples_post
    print(f"Epoch length (samples): pre={samples_pre} post={samples_post} total={epoch_len}")

    epochs_by_class = {0: [], 1: [], 2: []}
    skipped = 0

    for (t_event, ev_code, desc) in events:
        if ev_code not in epochs_by_class:
            # ignore unknown labels
            continue
        # find nearest sample index
        idx = int(round((t_event - t_start) * sr))
        start = idx - samples_pre
        end = idx + samples_post
        if start < 0 or end > eeg_full.shape[1]:
            skipped += 1
            # try expanding tolerance: if event is just a bit outside, we can clip and pad
            # we'll skip for safety; but log it
            print(f"Epoch descartado (fuera rango): event {ev_code} t={t_event} idx={idx} start={start} end={end}")
            continue
        epoch = eeg_full[:, start:end]
        epochs_by_class[ev_code].append((epoch, t_event))

    print("Epochs extracted per class:")
    for k in epochs_by_class:
        print(f"  class {k}: {len(epochs_by_class[k])}")
    print("Skipped epochs:", skipped)

    # If no epochs recovered for any class, abort
    total_epochs = sum(len(v) for v in epochs_by_class.values())
    if total_epochs == 0:
        print("No epochs extraídos. Revisa pre/post y sincronización.")
        return

    # Compute TFR + ERD/ERS per epoch, average per class
    tfrs_by_class = {0: [], 1: [], 2: []}
    f_ref = None
    t_ref = None

    for cls in epochs_by_class:
        for (epoch, t_event) in epochs_by_class[cls]:
            # epoch shape (nch, epoch_len)
            f, t_stft, tfr = compute_tfr(epoch, sr, nperseg=args.nperseg)  # (nch, nfreq, nt)
            # Align t_stft so that zero point corresponds to event time:
            # stft t runs from 0..T_epoch where 0 corresponds to epoch start = -pre
            t_eventlocked = t_stft - pre  # so that 0 == event
            # average channels
            tfr_mean_ch = np.mean(tfr, axis=0)  # (nfreq, nt)
            # compute ERD/ERS
            erd = erd_ers_percent(tfr_mean_ch, t_eventlocked, baseline_time_window=(-pre, 0.0))
            if erd is None:
                continue
            tfrs_by_class[cls].append((erd, f, t_eventlocked))
            f_ref = f
            t_ref = t_eventlocked

    # Average ERD per class (freq x time)
    erd_mean_by_class = {}
    for cls in tfrs_by_class:
        if len(tfrs_by_class[cls]) == 0:
            erd_mean_by_class[cls] = None
            continue
        arrs = np.array([it[0] for it in tfrs_by_class[cls]])  # (n_epochs, nfreq, nt)
        erd_mean = np.mean(arrs, axis=0)
        erd_mean_by_class[cls] = erd_mean

    # Plotting
    classes = {0: "Reposo", 1: "Abrir", 2: "Cerrar"}
    for cls in [0,1,2]:
        erd_mean = erd_mean_by_class.get(cls)
        if erd_mean is None:
            print(f"Clase {cls} no tiene ERD calculado. Saltando plot.")
            continue
        plt.figure(figsize=(10,5))
        plt.title(f"ERD/ERS (%) - {classes[cls]} (avg across epochs)")
        # erd_mean shape (nfreq, nt)
        plt.pcolormesh(t_ref, f_ref, erd_mean, shading='auto', cmap='RdBu_r')
        plt.colorbar(label='ERD/ERS (%)')
        plt.axvline(0.0, color='k', linestyle='--', label='Evento')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Frecuencia (Hz)')
        plt.ylim(1, min(60, f_ref.max()))
        plt.legend()
        plt.show()

    # Quick bandpower summary for chosen channel index
    ch_idx = args.chan_index
    print(f"\nBandpower summary on channel index {ch_idx} (0-based):")
    for cls in [0,1,2]:
        trials = epochs_by_class[cls]
        if len(trials) == 0:
            print(f"  class {cls}: 0 trials")
            continue
        mu_vals = []
        beta_vals = []
        for (epoch, _) in trials:
            ch = epoch[ch_idx, :]
            f, Pxx = np.fft.rfftfreq(len(ch), 1.0/sr), None
            # use STFT/PSD via welch for bandpower
            from scipy.signal import welch
            f_w, Pxx = welch(ch, fs=sr, nperseg=min(256, len(ch)))
            mu_idx = (f_w >= 8) & (f_w <= 12)
            beta_idx = (f_w >= 13) & (f_w <= 30)
            mu_vals.append(np.trapz(Pxx[mu_idx], f_w[mu_idx]))
            beta_vals.append(np.trapz(Pxx[beta_idx], f_w[beta_idx]))
        print(f"  class {cls}: trials {len(trials)}  mu_mean {np.mean(mu_vals):.4e}  beta_mean {np.mean(beta_vals):.4e}")

    print("\nDone.")

if __name__ == "__main__":
    main()


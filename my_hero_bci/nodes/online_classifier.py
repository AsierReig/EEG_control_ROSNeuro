#!/usr/bin/env python3
"""
online_classifier_3class.py
Nodo ROS para clasificación online (3 clases).
Publica /decision (String con "rest"/"open"/"close"), /decision_proba (Float32MultiArray)
y /features (Float32MultiArray).
"""

import rospy
import numpy as np
import joblib
from std_msgs.msg import Float32MultiArray, String
from rosneuro_msgs.msg import NeuroFrame
from scipy.signal import welch, butter, filtfilt
from collections import deque
import threading

# mapa de clases (entrenamiento usa 0,1,2)
CLASS_MAP = {0: "rest", 1: "open", 2: "close"}

def bandpass_filter(epoch, lowcut=8.0, highcut=30.0, fs=256, order=4):
    b, a = butter(order, [lowcut/(0.5*fs), highcut/(0.5*fs)], btype='band')
    return filtfilt(b, a, epoch, axis=1)

def extract_bandpower_from_window(epoch, fs=256, bands=[(8,12),(13,30)]):
    feats = []
    for ch in range(epoch.shape[0]):
        f, Pxx = welch(epoch[ch, :], fs=fs, nperseg=min(256, epoch.shape[1]))
        for lo, hi in bands:
            idx = (f>=lo) & (f<=hi)
            bp = np.trapz(Pxx[idx], f[idx]) if np.any(idx) else 1e-12
            feats.append(np.log(bp + 1e-12))
    return np.array(feats, dtype=np.float32)

class OnlineClassifierNode:
    def __init__(self):
        rospy.init_node('online_classifier')
        # params
        model_file = rospy.get_param('~model_path', '/home/tartaria/catkin_ws/src/my_hero_bci/models/model1.joblib')
        self.win_sec = float(rospy.get_param('~win_sec', 1.0))
        self.step_sec = float(rospy.get_param('~step_sec', 0.5))
        self.bands = rospy.get_param('~bands', [[8,12],[13,30]])
        self.proba_thresh = float(rospy.get_param('~proba_thresh', 0.6))
        self.debounce_count = int(rospy.get_param('~debounce_count', 3))
        self.wanted_channels = rospy.get_param('~wanted_channels', ['eeg:4','eeg:5','eeg:6'])
        self.topic_in = rospy.get_param('~topic_in', '/neurodata')

        data = joblib.load(model_file)
        # data expected: {'model': clf, 'channels': selected_labels, 'fs': fs}
        self.clf = data['model'] if isinstance(data, dict) and 'model' in data else data
        self.scaler = data.get('scaler', None) if isinstance(data, dict) else None
        self.trained_channels = data.get('channels', None) if isinstance(data, dict) else None
        self.trained_fs = data.get('fs', None) if isinstance(data, dict) else None

        # internal
        self.sr = None
        self.nchan = None
        self.win_samples = None
        self.step_samples = None
        self.buffer = None  # shape (nchan, current_samples)
        self.lock = threading.Lock()
        self.hist = deque(maxlen=self.debounce_count)

        # pubs/subs
        self.pub_decision = rospy.Publisher('/decision', String, queue_size=1)
        self.pub_proba = rospy.Publisher('/decision_proba', Float32MultiArray, queue_size=1)
        self.pub_features = rospy.Publisher('/features', Float32MultiArray, queue_size=1)

        rospy.Subscriber(self.topic_in, NeuroFrame, self.callback, queue_size=1)
        rospy.loginfo("Online classifier listo. Esperando /neurodata ...")
        rospy.spin()

    def callback(self, msg):
        # obtener sr (msg.sr según tu formato)
        if self.sr is None:
            try:
                self.sr = int(msg.sr)
            except:
                self.sr = int(self.trained_fs) if self.trained_fs is not None else 256
            self.win_samples = int(self.win_sec * self.sr)
            self.step_samples = int(self.step_sec * self.sr)
            rospy.loginfo("SR detectado: %d - win_samples=%d step=%d", self.sr, self.win_samples, self.step_samples)

        ns = msg.eeg.info.nsamples
        nc = msg.eeg.info.nchannels
        if ns == 0 or nc == 0:
            return

        data = np.array(msg.eeg.data, dtype=np.float32)
        # reshape correcto: (nc, ns)
        try:
            samples = data.reshape((nc, ns))
        except:
            samples = data.reshape((ns, nc)).T

        # seleccionar índices de canales entrenados (por label) o por wanted_channels param
        labels = [str(x) for x in msg.eeg.info.labels]
        # decide canales a usar
        if self.trained_channels:
            # map trained labels -> indices by substring
            ch_idx = []
            for tch in self.trained_channels:
                for i, lab in enumerate(labels):
                    if tch.lower() in lab.lower():
                        ch_idx.append(i)
                        break
        else:
            ch_idx = [i for i, lab in enumerate(labels) if any(w.lower() in lab.lower() for w in self.wanted_channels)]

        if not ch_idx:
            rospy.logwarn_throttle(10, "No se encontraron canales a usar en NeuroFrame. Labels disponibles: %s", labels)
            return

        samples = samples[ch_idx, :]  # (n_used_chan, ns)

        with self.lock:
            if self.buffer is None:
                self.buffer = samples.copy()
            else:
                # concatenar temporalmente por columnas (axis=1)
                self.buffer = np.hstack([self.buffer, samples])
                # opcional: mantener solo una ventana máxima para no crecer indefinidamente
                if self.buffer.shape[1] > max(self.win_samples*10, int(self.sr*60)):
                    # recortar manteniendo las últimas muestras
                    self.buffer = self.buffer[:, -self.win_samples*10:]

            # si no hay suficientes muestras para una ventana, salir
            if self.buffer.shape[1] < self.win_samples:
                return

            # tomar la última ventana completa
            window = self.buffer[:, -self.win_samples:]
            # filtrar (retorna shape (nchan, win_samples))
            try:
                window_f = bandpass_filter(window, fs=self.sr)
            except Exception as e:
                rospy.logwarn_throttle(5, "Error filtrado: %s", e)
                return

            # extraer features (debe coincidir con entrenamiento)
            feats = extract_bandpower_from_window(window_f, fs=self.sr, bands=self.bands)
            # apply scaler
            if self.scaler is not None:
                try:
                    feats = self.scaler.transform(feats.reshape(1, -1)).ravel()
                except Exception:
                    feats = feats
                    
            # sanity check dimension
            n_expected = len(self.trained_channels) * len(self.bands) if self.trained_channels else len(ch_idx) * len(self.bands)
            if feats.size != n_expected:
                rospy.logwarn_throttle(5, "Features dim mismatch: got %d expected %d", feats.size, n_expected)
                # intentar adaptar si posible (recortar o pad)
                if feats.size < n_expected:
                    feats = np.pad(feats, (0, n_expected - feats.size), 'constant', constant_values=(0,))
                else:
                    feats = feats[:n_expected]

            # publicar features
            self.pub_features.publish(Float32MultiArray(data=feats.tolist()))

            # predecir
            try:
                proba = self.clf.predict_proba(feats.reshape(1, -1))[0]
            except Exception as e:
                rospy.logwarn_throttle(5, "Error predicción: %s", e)
                return

            self.pub_proba.publish(Float32MultiArray(data=proba.tolist()))
            idx = int(np.argmax(proba))
            label_text = CLASS_MAP.get(idx, str(idx))
            self.hist.append((idx, label_text, proba[idx]))

            # debounce: exigir concordancia en hist y umbral de prob
            if len(self.hist) == self.debounce_count:
                ids = [h[0] for h in self.hist]
                topid = ids[0]
                if all([i == topid for i in ids]) and self.hist[-1][2] >= self.proba_thresh:
                    # publicar decisión textual
                    self.pub_decision.publish(String(data=CLASS_MAP.get(topid, str(topid))))
                    rospy.loginfo_throttle(2, "Decision final: %s p=%.2f", CLASS_MAP.get(topid, str(topid)), self.hist[-1][2])

if __name__ == '__main__':
    try:
        OnlineClassifierNode()
    except rospy.ROSInterruptException:
        pass

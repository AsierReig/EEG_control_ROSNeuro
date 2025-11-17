#!/usr/bin/env python3
"""
online_classifier_rosneuro.py
Adaptado para mensajes rosneuro_msgs/NeuroFrame con selección de canales.
"""

import rospy
import numpy as np
import joblib
from std_msgs.msg import Float32MultiArray, String
from rosneuro_msgs.msg import NeuroFrame
from scipy.signal import welch
from collections import deque
import threading

class OnlineClassifier:
    def __init__(self):
        rospy.init_node('online_classifier', anonymous=False)

        # === Parámetros configurables ===
        self.model_path     = rospy.get_param('~model_path', '/home/tartaria/catkin_ws/src/my_hero_bci/models/model1.joblib')
        self.sr             = float(rospy.get_param('~sr', 256.0))
        self.win_sec        = float(rospy.get_param('~win_sec', 1.0))
        self.step_sec       = float(rospy.get_param('~step_sec', 0.5))
        self.bands          = rospy.get_param('~bands', [[8,12], [13,30]])
        self.proba_thresh   = float(rospy.get_param('~proba_thresh', 0.7))
        self.debounce_count = int(rospy.get_param('~debounce_count', 3))
        self.topic_in       = rospy.get_param('~topic_in', '/neurodata')
        self.topic_features = rospy.get_param('~topic_features', '/features')
        self.wanted_channels = rospy.get_param('~wanted_channels', ['eeg:4','eeg:5','eeg:6'])  # canales usados en el entrenamiento

        # === Modelo ===
        self.model = joblib.load(self.model_path)
        self.classes_ = getattr(self.model, 'classes_', ['class0', 'class1'])

        # === Buffers ===
        self.nchan = None
        self.win_samples = None
        self.step_samples = None
        self.buffer = None
        self.buf_count = 0
        self.lock = threading.Lock()
        self.hist = deque(maxlen=self.debounce_count)

        # === Publicadores ===
        self.pub_decision = rospy.Publisher('/decision', String, queue_size=1)
        self.pub_proba    = rospy.Publisher('/decision_proba', Float32MultiArray, queue_size=1)
        self.pub_features = rospy.Publisher(self.topic_features, Float32MultiArray, queue_size=1)

        # === Suscripción ===
        rospy.Subscriber(self.topic_in, NeuroFrame, self.msg_cb, queue_size=1)
        rospy.loginfo("OnlineClassifier listo. Subscrito a %s", self.topic_in)
        rospy.spin()

    # -------------------- CALLBACK --------------------
    def msg_cb(self, msg):
        """Procesa un NeuroFrame con EEG"""
        eeg_data = np.array(msg.eeg.data, dtype=np.float32)
        nch = msg.eeg.info.nchannels
        nsamp = msg.eeg.info.nsamples
        if nch == 0 or nsamp == 0 or eeg_data.size == 0:
            return

        # Selección de canales
        ch_idx = [i for i, ch in enumerate(msg.eeg.info.labels) if ch in self.wanted_channels]
        if not ch_idx:
            rospy.logwarn_throttle(5, "No se encontraron los canales deseados en el NeuroFrame")
            return
        samples = eeg_data.reshape(nsamp, nch).T  # (nchan, nsamp)
        samples = samples[ch_idx, :]

        # Inicialización buffer
        if self.nchan is None:
            self.nchan = samples.shape[0]
            self.sr = msg.sr
            self.win_samples = int(self.win_sec * self.sr)
            self.step_samples = int(self.step_sec * self.sr)
            self.buffer = np.zeros((self.nchan, self.win_samples), dtype=np.float32)
            rospy.loginfo("EEG config detectada: %d canales, SR=%d Hz", self.nchan, self.sr)

        with self.lock:
            n_new = samples.shape[1]
            if n_new >= self.win_samples:
                self.buffer[:, :] = samples[:, -self.win_samples:]
                self.buf_count = self.win_samples
            else:
                self.buffer = np.roll(self.buffer, -n_new, axis=1)
                self.buffer[:, -n_new:] = samples
                self.buf_count = min(self.win_samples, self.buf_count + n_new)

            if self.buf_count >= self.win_samples:
                feats = self.compute_features(self.buffer)
                self.pub_features.publish(Float32MultiArray(data=feats.tolist()))
                self.do_predict(feats)

    # -------------------- FEATURES --------------------
    def compute_features(self, data_win):
        feats = []
        for ch in range(data_win.shape[0]):
            f, Pxx = welch(data_win[ch, :], fs=self.sr, nperseg=min(256, self.win_samples))
            for (lo, hi) in self.bands:
                idx = np.logical_and(f >= lo, f <= hi)
                bp = np.trapz(Pxx[idx], f[idx]) if np.any(idx) else 1e-12
                feats.append(np.log(bp + 1e-12))
        return np.array(feats, dtype=np.float32)

    # -------------------- PREDICCIÓN --------------------
    def do_predict(self, feats):
        x = feats.reshape(1, -1)
        try:
            proba = self.model.predict_proba(x)[0]
        except Exception as e:
            rospy.logwarn_throttle(5, "Error predicción: %s", str(e))
            return

        self.pub_proba.publish(Float32MultiArray(data=proba.tolist()))
        idx = int(np.argmax(proba))
        label = str(self.classes_[idx])
        self.hist.append(label)

        if len(self.hist) == self.debounce_count and all([l == label for l in self.hist]):
            if proba[idx] >= self.proba_thresh:
                self.pub_decision.publish(String(data=label))
                rospy.loginfo_throttle(2, "Decision: %s (p=%.2f)", label, proba[idx])


if __name__ == '__main__':
    try:
        OnlineClassifier()
    except rospy.ROSInterruptException:
        pass

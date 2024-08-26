import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import resampy
import soundfile as sf

class AudioAnalyzer:
    def __init__(self):
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')

    def analyze_audio(self, file_path):
        waveform, sample_rate = sf.read(file_path)
        waveform = resampy.resample(waveform, sample_rate, 16000)
        scores, embeddings, spectrogram = self.model(waveform)
        class_scores = tf.reduce_mean(scores, axis=0)
        top_classes = tf.argsort(class_scores, direction='DESCENDING')[:5]
        return top_classes.numpy()
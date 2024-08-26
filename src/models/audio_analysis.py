import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class AudioAnalyzer:
    def __init__(self):
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        self.cat_sound_indices = {
            'meow': 300,
            'purr': 301,
            'hiss': 302,
            'growl': 303
        }
        self.threshold = 0.5

    def process_audio(self, waveform):
        waveform = np.array(waveform, dtype=np.float32)
        scores, embeddings, spectrogram = self.yamnet_model(waveform)
        return scores, embeddings, spectrogram

    def interpret_cat_sounds(self, scores):
        detected_sounds = []
        for sound, index in self.cat_sound_indices.items():
            if scores[index] > self.threshold:
                detected_sounds.append(sound)
        return detected_sounds

    def analyze_vocalization(self, waveform):
        scores, embeddings, _ = self.process_audio(waveform)
        detected_sounds = self.interpret_cat_sounds(scores)
        
        if not detected_sounds:
            return "No cat sounds detected"
        
        mood = self.interpret_mood(detected_sounds)
        intensity = self.calculate_intensity(scores)
        duration = self.estimate_duration(waveform)
        
        return {
            "detected_sounds": detected_sounds,
            "mood": mood,
            "intensity": intensity,
            "duration": duration
        }

    def interpret_mood(self, detected_sounds):
        if 'purr' in detected_sounds:
            return "Content"
        elif 'meow' in detected_sounds:
            return "Communicative"
        elif 'hiss' in detected_sounds or 'growl' in detected_sounds:
            return "Agitated"
        else:
            return "Neutral"

    def calculate_intensity(self, scores):
        relevant_scores = [scores[i] for i in self.cat_sound_indices.values()]
        max_score = max(relevant_scores)
        if max_score < 0.3:
            return "Low"
        elif max_score < 0.7:
            return "Medium"
        else:
            return "High"

    def estimate_duration(self, waveform):
        # Assuming waveform is sampled at 16kHz
        duration_seconds = len(waveform) / 16000
        return f"{duration_seconds:.2f} seconds"
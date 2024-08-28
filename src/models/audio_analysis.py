import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import logging
from typing import List, Dict, Tuple, Union, Any
import librosa
import cv2
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioAnalyzer:
    def __init__(self):
        try:
            self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
            logger.info("YAMNet model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YAMNet model: {e}")
            raise

        self.cat_sound_indices = {
            'meow': 300,
            'purr': 301,
            'hiss': 302,
            'growl': 303
        }
        self.threshold = 0.5

    def process_audio(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            waveform = np.array(waveform, dtype=np.float32)
            scores, embeddings, spectrogram = self.yamnet_model(waveform)
            return scores, embeddings, spectrogram
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            raise

    def interpret_cat_sounds(self, scores: np.ndarray) -> List[str]:
        try:
            detected_sounds = []
            for sound, index in self.cat_sound_indices.items():
                if scores[index] > self.threshold:
                    detected_sounds.append(sound)
            return detected_sounds
        except Exception as e:
            logger.error(f"Error interpreting cat sounds: {e}")
            raise

    def analyze_vocalization(self, waveform: np.ndarray) -> Dict[str, Union[List[str], str, float]]:
        try:
            scores, embeddings, _ = self.process_audio(waveform)
            detected_sounds = self.interpret_cat_sounds(scores)
            
            if not detected_sounds:
                return {"message": "No cat sounds detected"}
            
            mood = self.interpret_mood(detected_sounds, scores)
            intensity = self.calculate_intensity(scores)
            duration = self.estimate_duration(waveform)
            
            return {
                "detected_sounds": detected_sounds,
                "mood": mood,
                "intensity": intensity,
                "duration": duration
            }
        except Exception as e:
            logger.error(f"Error in analyze_vocalization: {e}")
            return {"error": str(e)}

    def interpret_mood(self, detected_sounds: List[str], scores: np.ndarray) -> str:
        try:
            mood_scores = {
                "Content": 0,
                "Communicative": 0,
                "Agitated": 0,
                "Playful": 0
            }

            for sound in detected_sounds:
                if sound == 'purr':
                    mood_scores["Content"] += scores[self.cat_sound_indices[sound]] * 2
                elif sound == 'meow':
                    mood_scores["Communicative"] += scores[self.cat_sound_indices[sound]]
                    mood_scores["Playful"] += scores[self.cat_sound_indices[sound]] * 0.5
                elif sound in ['hiss', 'growl']:
                    mood_scores["Agitated"] += scores[self.cat_sound_indices[sound]] * 1.5

            dominant_mood = max(mood_scores, key=mood_scores.get)
            confidence = mood_scores[dominant_mood] / sum(mood_scores.values())

            return f"{dominant_mood} (confidence: {confidence:.2f})"
        except Exception as e:
            logger.error(f"Error interpreting mood: {e}")
            return "Mood interpretation error"

    def calculate_intensity(self, scores: np.ndarray) -> str:
        try:
            relevant_scores = [scores[i] for i in self.cat_sound_indices.values()]
            max_score = max(relevant_scores)
            if max_score < 0.3:
                return "Low"
            elif max_score < 0.7:
                return "Medium"
            else:
                return "High"
        except Exception as e:
            logger.error(f"Error calculating intensity: {e}")
            return "Intensity calculation error"

    def estimate_duration(self, waveform: np.ndarray) -> str:
        try:
            duration_seconds = len(waveform) / 16000
            return f"{duration_seconds:.2f} seconds"
        except Exception as e:
            logger.error(f"Error estimating duration: {e}")
            return "Duration estimation error"

    def calibrate(self, known_samples: Dict[str, List[np.ndarray]]) -> None:
        logger.info("Starting calibration...")
        try:
            new_thresholds = {sound: [] for sound in self.cat_sound_indices}

            for sound, samples in known_samples.items():
                for sample in samples:
                    scores, _, _ = self.process_audio(sample)
                    new_thresholds[sound].append(scores[self.cat_sound_indices[sound]])

            for sound, thresholds in new_thresholds.items():
                if thresholds:
                    self.cat_sound_indices[sound] = np.mean(thresholds) * 0.8  # 80% of mean as new threshold
                    logger.info(f"New threshold for {sound}: {self.cat_sound_indices[sound]}")

            logger.info("Calibration completed")
        except Exception as e:
            logger.error(f"Error during calibration: {e}")

    def load_audio(self, file_path: str, target_sr: int = 16000) -> np.ndarray:
        try:
            audio, sr = librosa.load(file_path, sr=None)
            if sr != target_sr:
                audio = librosa.resample(audio, sr, target_sr)
            return audio
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise

class VisualAnalyzer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface_extended.xml')
        self.body_model = MobileNetV2(weights='imagenet', include_top=False)
        
        # Load pre-trained models (you would need to train these models separately)
        try:
            self.ear_model = load_model('path_to_ear_position_model.h5')
            self.eye_model = load_model('path_to_eye_state_model.h5')
            self.posture_model = load_model('path_to_posture_model.h5')
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def detect_cat(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return len(faces) > 0, faces

    def analyze_ears(self, image, face):
        x, y, w, h = face
        face_img = image[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (128, 128))
        face_img = img_to_array(face_img)
        face_img = np.expand_dims(face_img, axis=0)
        face_img = preprocess_input(face_img)
        
        prediction = self.ear_model.predict(face_img)
        ear_positions = ['forward', 'neutral', 'backward']
        return ear_positions[np.argmax(prediction)]

    def analyze_eyes(self, image, face):
        x, y, w, h = face
        face_img = image[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (128, 128))
        face_img = img_to_array(face_img)
        face_img = np.expand_dims(face_img, axis=0)
        face_img = preprocess_input(face_img)
        
        prediction = self.eye_model.predict(face_img)
        eye_states = ['open', 'partially_open', 'closed']
        return eye_states[np.argmax(prediction)]

    def analyze_body_posture(self, image):
        resized = cv2.resize(image, (224, 224))
        img_array = img_to_array(resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        features = self.body_model.predict(img_array)
        flattened_features = features.flatten()
        
        prediction = self.posture_model.predict(np.expand_dims(flattened_features, axis=0))
        postures = ['relaxed', 'alert', 'aggressive', 'fearful']
        return postures[np.argmax(prediction)]

    def analyze_color(self, image):
        average_color = np.mean(image, axis=(0, 1))
        return average_color.tolist()

    def analyze_image(self, image_path: str) -> dict:
        try:
            image = self.preprocess_image(image_path)
            cat_detected, faces = self.detect_cat(image)
            
            if not cat_detected:
                return {"cat_detected": False, "message": "No cat detected in the image"}
            
            result = {
                "cat_detected": True,
                "num_cats": len(faces),
                "average_color": self.analyze_color(image),
                "body_posture": self.analyze_body_posture(image)
            }
            
            # Analyze the first detected cat face
            if len(faces) > 0:
                face = faces[0]
                result["ear_position"] = self.analyze_ears(image, face)
                result["eye_state"] = self.analyze_eyes(image, face)
            
            return result
        
        except Exception as e:
            logger.error(f"Error in visual analysis: {e}")
            return {"error": str(e)}
        
class CatBehaviorAnalyzer:
    def __init__(self):
        self.audio_analyzer = AudioAnalyzer()
        self.visual_analyzer = VisualAnalyzer()

    def analyze_behavior(self, audio_path: str, image_path: str = None) -> Dict[str, Any]:
        try:
            audio = self.audio_analyzer.load_audio(audio_path)
            audio_result = self.audio_analyzer.analyze_vocalization(audio)
            
            visual_result = {}
            if image_path:
                visual_result = self.visual_analyzer.analyze_image(image_path)

            return {
                "audio_analysis": audio_result,
                "visual_analysis": visual_result,
                "combined_interpretation": self.combine_analyses(audio_result, visual_result)
            }
        except Exception as e:
            logger.error(f"Error in analyze_behavior: {e}")
            return {"error": str(e)}

    def combine_analyses(self, audio_result: Dict[str, Any], visual_result: Dict[str, Any]) -> str:
        try:
            if "error" in audio_result or "error" in visual_result:
                return "Unable to perform combined analysis due to errors in individual analyses."

            mood = audio_result.get("mood", "").split()[0]  # Get the mood without confidence
            intensity = audio_result.get("intensity", "")
            detected_sounds = audio_result.get("detected_sounds", [])

            cat_detected = visual_result.get("cat_detected", False)
            ear_position = visual_result.get("ear_position", "")
            eye_state = visual_result.get("eye_state", "")
            body_posture = visual_result.get("body_posture", "")
            num_cats = visual_result.get("num_cats", 0)

            interpretation = ""

            if not cat_detected:
                interpretation += "No cat visually detected, but audio analysis was performed. "
            else:
                interpretation += f"{num_cats} cat(s) visually detected. "

            interpretation += f"Audio analysis suggests the cat is {mood.lower()} "
            interpretation += f"with {intensity.lower()} intensity. "

            if cat_detected:
                interpretation += f"Visual analysis shows: ear position is {ear_position}, "
                interpretation += f"eyes are {eye_state}, and body posture is {body_posture}. "

            # Combine audio and visual cues for a more comprehensive interpretation
            if "purr" in detected_sounds and ear_position == "forward" and body_posture == "relaxed":
                interpretation += "The cat appears to be very content and relaxed. "
            elif "meow" in detected_sounds and eye_state == "open" and body_posture == "alert":
                interpretation += "The cat seems to be trying to communicate and is alert. "
            elif ("hiss" in detected_sounds or "growl" in detected_sounds) and ear_position == "backward" and body_posture == "aggressive":
                interpretation += "The cat may be feeling threatened or aggressive. "
            elif mood == "Content" and body_posture == "relaxed":
                interpretation += "The cat seems to be in a good mood and relaxed. "
            elif mood == "Agitated" and (body_posture == "alert" or body_posture == "aggressive"):
                interpretation += "The cat appears to be stressed or anxious. "
            
            interpretation += "Please note that this interpretation is based on both audio and visual cues, "
            interpretation += "but may not capture all nuances of cat behavior. For accurate behavioral assessment, "
            interpretation += "consult with a veterinary behaviorist."

            return interpretation

        except Exception as e:
            logger.error(f"Error in combining analyses: {e}")
            return "Error occurred while combining audio and visual analyses."

# Example usage
if __name__ == "__main__":
    analyzer = CatBehaviorAnalyzer()
    
    try:
        result = analyzer.analyze_behavior("path/to/cat_audio.wav", "path/to/cat_image.jpg")
        print("Combined Analysis Result:")
        print(json.dumps(result, indent=2))
        
        print("\nAudio Analysis:")
        print(json.dumps(result['audio_analysis'], indent=2))
        
        print("\nVisual Analysis:")
        print(json.dumps(result['visual_analysis'], indent=2))
        
        print("\nCombined Interpretation:")
        print(result['combined_interpretation'])
        
    except Exception as e:
        print(f"An error occurred: {e}")

    # Example of calibration (you would need actual cat sound samples for this)
    # known_samples = {
    #     "meow": [analyzer.audio_analyzer.load_audio("path/to/meow1.wav"), analyzer.audio_analyzer.load_audio("path/to/meow2.wav")],
    #     "purr": [analyzer.audio_analyzer.load_audio("path/to/purr1.wav"), analyzer.audio_analyzer.load_audio("path/to/purr2.wav")],
    # }
    # analyzer.audio_analyzer.calibrate(known_samples)
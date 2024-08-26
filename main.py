import cv2
import numpy as np
import librosa
from fastapi import FastAPI, File, UploadFile
from io import BytesIO

from audio_analyzer import AudioAnalyzer
from cat_face_detector import CatFaceDetector
from mood_interpreter import MoodInterpreter

app = FastAPI()

class Purrception:
    def __init__(self):
        self.audio_analyzer = AudioAnalyzer()
        self.face_detector = CatFaceDetector('/workspace/Purrception/data/haarcascade_frontalcatface.xml')
        self.mood_interpreter = MoodInterpreter()

    def analyze_cat(self, audio_waveform, image):
        # Analyze audio
        audio_result = self.audio_analyzer.analyze_vocalization(audio_waveform)

        # Analyze image
        face_analysis = self.face_detector.analyze_face(image)

        # Interpret mood
        if face_analysis["detected"]:
            face_result = face_analysis["faces"][0]["expression"]
        else:
            face_result = "No face detected"

        mood_analysis = self.mood_interpreter.interpret_cat_mood(
            audio_result["detected_sounds"][0] if audio_result["detected_sounds"] else "No sound",
            "Relaxed",  # Placeholder for pose analysis
            face_result
        )

        return {
            "audio_analysis": audio_result,
            "face_analysis": face_analysis,
            "mood_analysis": mood_analysis
        }

purrception = Purrception()

def audio_to_numpy(audio_file):
    audio = AudioSegment.from_file(audio_file)
    samples = audio.get_array_of_samples()
    return np.array(samples).astype(np.float32) / 32768.0  # Normalize to [-1.0, 1.0]

@app.post("/analyze")
async def analyze_cat(audio: UploadFile = File(...), image: UploadFile = File(...)):
    # Process audio
    audio_content = await audio.read()
    audio_np = audio_to_numpy(BytesIO(audio_content))
    
    # Process image
    image_content = await image.read()
    image_np = cv2.imdecode(np.frombuffer(image_content, np.uint8), cv2.IMREAD_COLOR)

    # Analyze using Purrception
    results = purrception.analyze_cat(audio_np, image_np)
    
    return results

@app.get("/")
async def root():
    return {"message": "Welcome to Purrception API"}

if __name__ == "__main__":
    # Test audio analysis
    test_audio_path = '/workspace/Purrception/data/test_audio.wav'
    test_waveform, _ = librosa.load(test_audio_path, sr=16000)
    audio_result = purrception.audio_analyzer.analyze_vocalization(test_waveform)
    print("Audio Analysis Result:", audio_result)

    # Test face detection
    test_image_path = '/workspace/Purrception/data/test_cat_image.jpg'
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        print(f"Failed to load image from {test_image_path}")
    else:
        face_analysis = purrception.face_detector.analyze_face(test_image)
        print("Face Analysis Result:", face_analysis)

        # Visualize face detection (if a face was detected)
        if face_analysis["detected"]:
            image_with_faces = purrception.face_detector.draw_faces(test_image, [face["position"] + (int(face["size"]**0.5),)*2 for face in face_analysis["faces"]])
            image_with_features = purrception.face_detector.draw_features(image_with_faces, [face["position"] + (int(face["size"]**0.5),)*2 for face in face_analysis["faces"]])
            cv2.imshow('Cat Face Detection', image_with_features)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Test mood interpretation
    test_audio_result = "Purr"
    test_pose_result = "Relaxed"
    test_face_result = "Slow blink"
    mood_analysis = purrception.mood_interpreter.interpret_cat_mood(test_audio_result, test_pose_result, test_face_result)
    print("Mood Analysis Result:", mood_analysis)

    print("\nMood Description:")
    print(purrception.mood_interpreter.get_mood_description(mood_analysis['mood'], mood_analysis['confidence']))
    print("\nRecommendations:")
    print(purrception.mood_interpreter.provide_recommendations(mood_analysis['mood']))

    # Start the FastAPI server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
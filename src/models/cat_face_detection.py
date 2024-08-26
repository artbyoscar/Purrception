import cv2
import numpy as np
from enum import Enum

class FaceFeatures(Enum):
    EYES = 1
    NOSE = 2
    MOUTH = 3

class CatFaceDetector:
    def __init__(self, cascade_path):
        self.cat_face_cascade = cv2.CascadeClassifier(cascade_path)
        self.eye_cascade = cv2.CascadeClassifier('path_to_eye_cascade.xml')
        self.nose_cascade = cv2.CascadeClassifier('path_to_nose_cascade.xml')

    def detect_cat_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cat_faces = self.cat_face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        return cat_faces

    def detect_features(self, image, face):
        x, y, w, h = face
        roi_gray = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        
        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        nose = self.nose_cascade.detectMultiScale(roi_gray)
        
        return {
            FaceFeatures.EYES: eyes,
            FaceFeatures.NOSE: nose
        }

    def analyze_face(self, image):
        faces = self.detect_cat_face(image)
        if len(faces) == 0:
            return {"detected": False, "message": "No cat face detected"}

        face_analyses = []
        for face in faces:
            features = self.detect_features(image, face)
            face_analysis = self.interpret_face(face, features)
            face_analyses.append(face_analysis)

        return {"detected": True, "faces": face_analyses}

    def interpret_face(self, face, features):
        x, y, w, h = face
        face_size = w * h
        aspect_ratio = w / h

        interpretation = {
            "position": (x, y),
            "size": face_size,
            "aspect_ratio": aspect_ratio,
            "orientation": self.estimate_orientation(aspect_ratio),
            "eyes_detected": len(features[FaceFeatures.EYES]) > 0,
            "nose_detected": len(features[FaceFeatures.NOSE]) > 0
        }

        interpretation["expression"] = self.estimate_expression(interpretation)
        return interpretation

    def estimate_orientation(self, aspect_ratio):
        if 0.9 <= aspect_ratio <= 1.1:
            return "Frontal"
        elif aspect_ratio < 0.9:
            return "Profile"
        else:
            return "Tilted"

    def estimate_expression(self, face_data):
        if face_data["eyes_detected"] and face_data["nose_detected"]:
            if face_data["orientation"] == "Frontal":
                return "Alert"
            elif face_data["orientation"] == "Tilted":
                return "Curious"
        elif face_data["eyes_detected"] and not face_data["nose_detected"]:
            return "Sleepy"
        else:
            return "Neutral"

    def draw_faces(self, image, faces):
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return image

    def draw_features(self, image, faces):
        for face in faces:
            x, y, w, h = face
            roi_color = image[y:y+h, x:x+w]
            features = self.detect_features(image, face)
            
            for eyes in features[FaceFeatures.EYES]:
                ex, ey, ew, eh = eyes
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            for nose in features[FaceFeatures.NOSE]:
                nx, ny, nw, nh = nose
                cv2.rectangle(roi_color, (nx, ny), (nx+nw, ny+nh), (0, 0, 255), 2)
        
        return image
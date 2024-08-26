from transformers import pipeline
from enum import Enum

class CatMood(Enum):
    HAPPY = "Happy"
    CONTENT = "Content"
    CURIOUS = "Curious"
    ANXIOUS = "Anxious"
    ANGRY = "Angry"
    FEARFUL = "Fearful"
    NEUTRAL = "Neutral"

class MoodInterpreter:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.mood_weights = {
            'audio': 0.4,
            'pose': 0.3,
            'face': 0.3
        }

    def analyze_sentiment(self, text):
        result = self.sentiment_analyzer(text)[0]
        return result['label'], result['score']

    def interpret_cat_mood(self, audio_result, pose_result, face_result):
        audio_mood = self.interpret_audio(audio_result)
        pose_mood = self.interpret_pose(pose_result)
        face_mood = self.interpret_face(face_result)

        weighted_mood = self.calculate_weighted_mood(audio_mood, pose_mood, face_mood)
        confidence = self.calculate_confidence(audio_mood, pose_mood, face_mood)

        return {
            "mood": weighted_mood,
            "confidence": confidence,
            "components": {
                "audio": audio_mood,
                "pose": pose_mood,
                "face": face_mood
            }
        }

    def interpret_audio(self, audio_result):
        # Map audio results to cat moods
        audio_mood_map = {
            "Meow": CatMood.CURIOUS,
            "Purr": CatMood.CONTENT,
            "Hiss": CatMood.ANGRY,
            "Growl": CatMood.FEARFUL
        }
        return audio_mood_map.get(audio_result, CatMood.NEUTRAL)

    def interpret_pose(self, pose_result):
        # Map pose results to cat moods
        pose_mood_map = {
            "Relaxed": CatMood.CONTENT,
            "Alert": CatMood.CURIOUS,
            "Arched back": CatMood.FEARFUL,
            "Tail up": CatMood.HAPPY
        }
        return pose_mood_map.get(pose_result, CatMood.NEUTRAL)

    def interpret_face(self, face_result):
        # Map face results to cat moods
        face_mood_map = {
            "Eyes wide": CatMood.CURIOUS,
            "Ears back": CatMood.ANXIOUS,
            "Pupils dilated": CatMood.FEARFUL,
            "Slow blink": CatMood.CONTENT
        }
        return face_mood_map.get(face_result, CatMood.NEUTRAL)

    def calculate_weighted_mood(self, audio_mood, pose_mood, face_mood):
        mood_scores = {mood: 0 for mood in CatMood}
        
        mood_scores[audio_mood] += self.mood_weights['audio']
        mood_scores[pose_mood] += self.mood_weights['pose']
        mood_scores[face_mood] += self.mood_weights['face']

        return max(mood_scores, key=mood_scores.get)

    def calculate_confidence(self, audio_mood, pose_mood, face_mood):
        if audio_mood == pose_mood == face_mood:
            return 1.0
        elif audio_mood == pose_mood or audio_mood == face_mood or pose_mood == face_mood:
            return 0.7
        else:
            return 0.4

    def get_mood_description(self, mood, confidence):
        descriptions = {
            CatMood.HAPPY: "Your cat seems happy and content. They might be purring or showing signs of affection.",
            CatMood.CONTENT: "Your cat appears relaxed and at ease. They're likely feeling comfortable in their environment.",
            CatMood.CURIOUS: "Your cat seems interested in something. They might be exploring or investigating their surroundings.",
            CatMood.ANXIOUS: "Your cat may be feeling a bit uneasy. They might need some reassurance or a quiet space.",
            CatMood.ANGRY: "Your cat appears agitated. It's best to give them some space and remove any potential stressors.",
            CatMood.FEARFUL: "Your cat seems scared or threatened. Try to identify the source of their fear and provide a safe, calm environment.",
            CatMood.NEUTRAL: "Your cat's mood is neutral. They're neither particularly excited nor upset."
        }

        confidence_level = "high" if confidence > 0.7 else "moderate" if confidence > 0.4 else "low"
        
        return f"{descriptions[mood]} (Confidence: {confidence_level})"

    def provide_recommendations(self, mood):
        recommendations = {
            CatMood.HAPPY: "Enjoy this moment with your cat. Consider engaging in play or offering treats to reinforce positive experiences.",
            CatMood.CONTENT: "This is a good time for gentle interaction or quiet companionship. Your cat might enjoy some soft petting.",
            CatMood.CURIOUS: "Encourage your cat's curiosity with interactive toys or new, safe objects to explore.",
            CatMood.ANXIOUS: "Create a calm environment. Provide a hiding spot and consider using pheromone diffusers to reduce anxiety.",
            CatMood.ANGRY: "Give your cat space and time to calm down. Ensure they have a safe, quiet place to retreat.",
            CatMood.FEARFUL: "Identify and remove the source of fear if possible. Speak softly and move slowly around your cat.",
            CatMood.NEUTRAL: "This is a good time for routine activities like feeding or gentle play, depending on your cat's preferences."
        }
        
        return recommendations[mood]
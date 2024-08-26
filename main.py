from fastapi import FastAPI, File, UploadFile
from audio_analyzer import analyze_audio
from video_analyzer import detect_cat_face
from sentiment_analyzer import analyze_sentiment

app = FastAPI()

@app.post("/analyze")
async def analyze_cat(audio: UploadFile, video: UploadFile):
    # Process audio and video
    audio_data = await audio.read()
    video_data = await video.read()
    
    # Analyze audio
    audio_results = analyze_audio(audio_data)
    
    # Analyze video
    video_frames = extract_frames(video_data)
    face_detections = [detect_cat_face(frame) for frame in video_frames]
    
    # Combine results and perform sentiment analysis
    combined_results = combine_analysis(audio_results, face_detections)
    sentiment = analyze_sentiment(combined_results)
    
    return {"results": combined_results, "sentiment": sentiment}
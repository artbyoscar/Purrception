import pytest
from cat_analyzer import CatBehaviorAnalyzer
import numpy as np

@pytest.fixture
def cat_behavior_analyzer():
    return CatBehaviorAnalyzer()

def test_combine_analyses(cat_behavior_analyzer):
    audio_result = {
        "mood": "Content",
        "intensity": "Medium",
        "detected_sounds": ["purr"]
    }
    visual_result = {
        "cat_detected": True,
        "ear_position": "forward",
        "eye_state": "open",
        "body_posture": "relaxed"
    }
    interpretation = cat_behavior_analyzer.combine_analyses(audio_result, visual_result)
    assert isinstance(interpretation, str)
    assert "content" in interpretation.lower()
    assert "relaxed" in interpretation.lower()

def test_analyze_behavior(cat_behavior_analyzer, tmp_path):
    # Create mock audio and image files
    mock_audio_path = tmp_path / "mock_cat_audio.wav"
    mock_image_path = tmp_path / "mock_cat_image.jpg"
    
    # Create dummy audio file
    np.random.rand(16000).astype(np.float32).tofile(mock_audio_path)
    
    # Create dummy image file
    mock_image = np.random.rand(300, 300, 3) * 255
    cv2.imwrite(str(mock_image_path), mock_image)

    result = cat_behavior_analyzer.analyze_behavior(str(mock_audio_path), str(mock_image_path))
    assert isinstance(result, dict)
    assert 'audio_analysis' in result
    assert 'visual_analysis' in result
    assert 'combined_interpretation' in result
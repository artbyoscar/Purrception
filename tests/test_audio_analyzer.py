import pytest
from cat_analyzer import AudioAnalyzer
import numpy as np

@pytest.fixture
def audio_analyzer():
    return AudioAnalyzer()

def test_process_audio(audio_analyzer):
    # Create a mock waveform
    mock_waveform = np.random.rand(16000)  # 1 second of random audio at 16kHz
    scores, embeddings, spectrogram = audio_analyzer.process_audio(mock_waveform)
    assert scores.shape[0] == 521  # YAMNet output size
    assert embeddings.shape[1] == 1024  # YAMNet embedding size

def test_interpret_cat_sounds(audio_analyzer):
    mock_scores = np.zeros(521)
    mock_scores[audio_analyzer.cat_sound_indices['meow']] = 0.7
    detected_sounds = audio_analyzer.interpret_cat_sounds(mock_scores)
    assert 'meow' in detected_sounds

def test_analyze_vocalization(audio_analyzer):
    mock_waveform = np.random.rand(16000)
    result = audio_analyzer.analyze_vocalization(mock_waveform)
    assert isinstance(result, dict)
    assert 'mood' in result
    assert 'intensity' in result
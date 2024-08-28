import pytest
from cat_analyzer import VisualAnalyzer
import numpy as np
import cv2

@pytest.fixture
def visual_analyzer():
    return VisualAnalyzer()

def test_detect_cat(visual_analyzer):
    # Create a mock image with a cat face
    mock_image = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(mock_image, (100, 100), (200, 200), (255, 255, 255), -1)  # White rectangle as "cat face"
    cat_detected, faces = visual_analyzer.detect_cat(mock_image)
    assert cat_detected
    assert len(faces) > 0

def test_analyze_image(visual_analyzer, tmp_path):
    # Create a temporary image file
    mock_image_path = tmp_path / "mock_cat.jpg"
    mock_image = np.random.rand(300, 300, 3) * 255
    cv2.imwrite(str(mock_image_path), mock_image)

    result = visual_analyzer.analyze_image(str(mock_image_path))
    assert isinstance(result, dict)
    assert 'cat_detected' in result
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from operators import roboflow_detection_op as rf_module
from operators.roboflow_detection_op import RoboflowHostedDetector


def test_parse_result_maps_classes_and_boxes():
    detector = object.__new__(RoboflowHostedDetector)
    detector.class_name_map = {
        "Gallbladder": "gallbladder",
        "Duct": "cystic_duct",
    }
    detector.target_classes = ["gallbladder", "cystic_duct"]
    detector.model_id = "laparoscopy/8"

    result = {
        "predictions": [
            {
                "x": 100,
                "y": 120,
                "width": 40,
                "height": 20,
                "class": "Gallbladder",
                "confidence": 0.9,
                "class_id": 1,
            },
            {
                "x": 60,
                "y": 80,
                "width": 20,
                "height": 10,
                "class": "Duct",
                "confidence": 0.8,
                "class_id": 2,
            },
            {
                "x": 10,
                "y": 10,
                "width": 5,
                "height": 5,
                "class": "Forceps",
                "confidence": 0.7,
                "class_id": 3,
            },
        ]
    }

    detections = RoboflowHostedDetector._parse_result(detector, result)
    assert len(detections) == 2
    assert detections[0].class_name == "gallbladder"
    assert detections[0].bbox == [80.0, 110.0, 120.0, 130.0]
    assert detections[1].class_name == "cystic_duct"


def test_empty_api_key_uses_environment_variable(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "env-key")

    class DummyClient:
        def __init__(self, api_url, api_key):
            self.api_url = api_url
            self.api_key = api_key

        def configure(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(rf_module, "InferenceHTTPClient", DummyClient)
    monkeypatch.setattr(rf_module, "InferenceConfiguration", lambda **kwargs: kwargs)

    detector = RoboflowHostedDetector(
        model_id="laparoscopy/14",
        api_key="",
        api_key_env="ROBOFLOW_API_KEY",
    )

    assert detector.api_key == "env-key"
    assert detector.client.api_key == "env-key"

"""
AI信号检测与调制识别模块
基于深度学习的信号检测和调制方式自动识别
"""

from .modulator import Modulator, SignalGenerator
from .detector import SignalDetector, DetectorTrainer
from .recognizer import CNNModulationRecognizer, RecognitionTrainer

# 别名
ModulationRecognizer = CNNModulationRecognizer

__all__ = [
    "Modulator",
    "SignalGenerator",
    "SignalDetector",
    "DetectorTrainer",
    "ModulationRecognizer",
    "CNNModulationRecognizer",
    "RecognitionTrainer",
]

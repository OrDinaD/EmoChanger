import os
import torch
import logging
from models.wav2vec2_model import Wav2Vec2EmotionModel
from utils.logger import setup_logger
from configs.config import EMOTION_LABELS

# Настраиваем логирование
logger = setup_logger()

def test_model():
    try:
        # Создаем директорию для моделей
        os.makedirs("data/models", exist_ok=True)
        
        # Инициализируем модель
        logger.info("Инициализация модели...")
        model = Wav2Vec2EmotionModel()
        
        # Сохраняем модель
        logger.info("Сохранение модели...")
        model.save_model("data/models/test_model.pth")
        
        # Загружаем модель
        logger.info("Загрузка модели...")
        model = Wav2Vec2EmotionModel(model_path="data/models/test_model.pth")
        
        # Создаем тестовый тензор
        test_input = torch.randn(1, 16000)  # 1 секунда аудио при 16 кГц
        
        # Тестируем предсказание
        logger.info("Тестирование предсказания...")
        emotion_idx, probs = model.predict_emotion(test_input)
        emotion_label = EMOTION_LABELS[emotion_idx]
        logger.info(f"Предсказанная эмоция: {emotion_label}")
        
        # Выводим вероятности для каждой эмоции
        for label, prob in zip(EMOTION_LABELS, probs):
            logger.info(f"{label}: {prob:.2%}")
        
        logger.info("Тестирование успешно завершено")
        
    except Exception as e:
        logger.error(f"Ошибка при тестировании: {str(e)}")
        raise

if __name__ == "__main__":
    test_model() 
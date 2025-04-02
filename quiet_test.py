import os
import sys
import torch
import logging
import traceback
from src.models.wav2vec2_model import Wav2Vec2EmotionModel
from src.preprocessing.audio_processor import AudioProcessor
from src.utils.dataset import EmotionDataset, create_dataset_csv
from src.configs.config import (
    SAMPLE_RATE, DATA_DIR, MODELS_DIR, RAW_DATA_DIR, 
    PROCESSED_DATA_DIR, MAX_AUDIO_LENGTH, BATCH_SIZE, 
    VALIDATION_SPLIT, LEARNING_RATE
)

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug.log', mode='w', encoding='utf-8')
    ]
)

logger = logging.getLogger("quiet_test")

def ensure_directories():
    """Создаем необходимые директории"""
    dirs = [DATA_DIR, MODELS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Директория существует: {directory}")
    return True

def test_model_initialization():
    """Тест инициализации модели"""
    try:
        logger.info("Инициализация модели...")
        model = Wav2Vec2EmotionModel()
        logger.info("Модель успешно инициализирована")
        return model
    except Exception as e:
        logger.error(f"Ошибка инициализации модели: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def test_model_save_load(model):
    """Тест сохранения и загрузки модели"""
    if not model:
        logger.error("Нельзя проверить сохранение модели: модель не инициализирована")
        return None
        
    try:
        # Сохраняем модель
        save_path = os.path.join(MODELS_DIR, "test_model.pth")
        logger.info(f"Сохранение модели в {save_path}...")
        model.save_model(save_path)
        logger.info("Модель успешно сохранена")
        
        # Загружаем модель
        logger.info("Загрузка модели...")
        loaded_model = Wav2Vec2EmotionModel(model_path=save_path)
        logger.info("Модель успешно загружена")
        
        return loaded_model
    except Exception as e:
        logger.error(f"Ошибка при сохранении/загрузке модели: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def test_audio_processing():
    """Тест обработки аудио"""
    try:
        logger.info("Инициализация аудио процессора...")
        audio_processor = AudioProcessor()
        logger.info("Аудио процессор инициализирован")
        
        # Создаем тестовый аудиофайл если его нет
        test_audio_dir = os.path.join(RAW_DATA_DIR)
        os.makedirs(test_audio_dir, exist_ok=True)
        
        test_audio_path = os.path.join(test_audio_dir, "test_audio.wav")
        if not os.path.exists(test_audio_path):
            logger.info("Создание тестового аудиофайла...")
            # Создаем тестовый аудиотензор
            waveform = torch.randn(1, SAMPLE_RATE)  # 1 секунда аудио
            audio_processor.save_audio(waveform, test_audio_path)
            logger.info(f"Тестовый аудиофайл создан: {test_audio_path}")
        
        # Загружаем тестовый аудиофайл
        logger.info(f"Загрузка аудиофайла: {test_audio_path}")
        waveform = audio_processor.load_audio(test_audio_path)
        logger.info(f"Аудиофайл загружен, размер: {waveform.shape}")
        
        # Предобработка аудио
        logger.info("Предобработка аудио...")
        processed_waveform = audio_processor.preprocess_audio(waveform)
        logger.info(f"Аудио предобработано, размер: {processed_waveform.shape}")
        
        # Извлечение признаков
        logger.info("Извлечение признаков...")
        features = audio_processor.extract_features(processed_waveform)
        logger.info(f"Признаки извлечены, размер: {features.shape}")
        
        return waveform
    except Exception as e:
        logger.error(f"Ошибка при обработке аудио: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def test_dataset_creation():
    """Тест создания датасета"""
    try:
        logger.info("Создание тестового датасета...")
        # Создаем CSV файл с метками
        test_csv_path = os.path.join(DATA_DIR, "test_dataset.csv")
        
        # Создаем файлы с разными эмоциями для теста
        emotions = ["neutral", "happy", "sad", "angry", "fear"]
        audio_processor = AudioProcessor()
        
        for i, emotion in enumerate(emotions):
            filename = f"{emotion}_test.wav"
            filepath = os.path.join(RAW_DATA_DIR, filename)
            if not os.path.exists(filepath):
                # Создаем тестовый аудиофайл
                waveform = torch.randn(1, SAMPLE_RATE)  # 1 секунда аудио
                audio_processor.save_audio(waveform, filepath)
                logger.info(f"Создан тестовый аудиофайл: {filepath}")
        
        create_dataset_csv(RAW_DATA_DIR, test_csv_path)
        logger.info(f"CSV файл создан: {test_csv_path}")
        
        # Создаем датасет
        dataset = EmotionDataset(RAW_DATA_DIR, test_csv_path)
        logger.info(f"Датасет создан, размер: {len(dataset)}")
        
        # Получаем первый элемент
        waveform, emotion_idx = dataset[0]
        logger.info(f"Первый элемент датасета: аудио {waveform.shape}, эмоция {emotion_idx}")
        
        return dataset
    except Exception as e:
        logger.error(f"Ошибка при создании датасета: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def test_model_prediction(model, waveform):
    """Тест предсказания эмоции"""
    if not model or waveform is None:
        logger.error("Нельзя проверить предсказание: модель или аудио не инициализированы")
        return False
        
    try:
        logger.info("Предсказание эмоции...")
        emotion_idx, probs = model.predict_emotion(waveform)
        logger.info(f"Предсказанная эмоция: {emotion_idx}")
        logger.info(f"Вероятности: {probs}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при предсказании эмоции: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_change_emotion(model, waveform):
    """Тест изменения эмоции в аудио"""
    if not model or waveform is None:
        logger.error("Нельзя проверить изменение эмоции: модель или аудио не инициализированы")
        return False
        
    try:
        logger.info("Изменение эмоции...")
        # Пробуем изменить на "счастливую" (индекс 1)
        modified_audio = model.change_emotion(waveform, 1)
        logger.info(f"Эмоция изменена, размер аудио: {modified_audio.shape}")
        
        # Сохраняем модифицированное аудио
        audio_processor = AudioProcessor()
        output_path = os.path.join(PROCESSED_DATA_DIR, "modified_audio.wav")
        audio_processor.save_audio(modified_audio, output_path)
        logger.info(f"Модифицированное аудио сохранено: {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Ошибка при изменении эмоции: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    logger.info("Начало тихого тестирования...")
    
    if not ensure_directories():
        return
    
    model = test_model_initialization()
    if model:
        model = test_model_save_load(model)
    
    waveform = test_audio_processing()
    
    dataset = test_dataset_creation()
    
    if model and waveform is not None:
        test_model_prediction(model, waveform)
        test_change_emotion(model, waveform)
    
    logger.info("Тестирование завершено.")

if __name__ == "__main__":
    main() 
import os
import pandas as pd
import torch
import logging
from torch.utils.data import Dataset
from ..preprocessing.audio_processor import AudioProcessor
from ..configs.config import EMOTION_LABELS

# Настраиваем логгер
logger = logging.getLogger(__name__)

class EmotionDataset(Dataset):
    def __init__(self, data_dir, csv_path):
        self.data_dir = data_dir
        self.audio_processor = AudioProcessor()
        
        try:
            # Загружаем метки из CSV
            logger.info(f"Загрузка датасета из {csv_path}")
            self.df = pd.read_csv(csv_path)
            logger.info(f"Загружено {len(self.df)} записей")
        except Exception as e:
            logger.error(f"Ошибка при загрузке датасета: {str(e)}")
            # Создаем пустой датафрейм
            self.df = pd.DataFrame(columns=['filename', 'emotion'])
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        try:
            # Получаем путь к аудиофайлу и метку эмоции
            filename = self.df.iloc[idx]['filename']
            emotion = self.df.iloc[idx]['emotion']
            
            # Загружаем и предобрабатываем аудио
            audio_path = os.path.join(self.data_dir, filename)
            logger.debug(f"Загрузка аудио: {audio_path}")
            
            if not os.path.exists(audio_path):
                logger.warning(f"Файл не найден: {audio_path}, использую случайный шум")
                # Создаем случайный шум вместо отсутствующего файла
                waveform = torch.randn(1, 16000)  # 1 секунда шума при 16 кГц
            else:
                waveform = self.audio_processor.load_audio(audio_path)
                
            waveform = self.audio_processor.preprocess_audio(waveform)
            
            # Получаем индекс эмоции из словаря
            # Используем прямое сопоставление для нескольких распространенных эмоций
            emotion_map = {label.lower(): i for i, label in enumerate(EMOTION_LABELS)}
            
            emotion_idx = emotion_map.get(emotion.lower(), 0)  # По умолчанию нейтральный
            logger.debug(f"Эмоция: {emotion} -> индекс: {emotion_idx}")
            
            return waveform, emotion_idx
        except Exception as e:
            logger.error(f"Ошибка при получении элемента датасета [{idx}]: {str(e)}")
            # В случае ошибки возвращаем заглушку
            return torch.randn(1, 16000), 0  # Случайный шум и нейтральная эмоция

def create_dataset_csv(data_dir, output_path):
    """Создание CSV файла с метками для датасета"""
    try:
        logger.info(f"Создание CSV файла для датасета в директории {data_dir}")
        data = []
        
        # Проходим по всем файлам в директории
        for filename in os.listdir(data_dir):
            if filename.lower().endswith('.wav'):
                # Пытаемся извлечь метку эмоции из имени файла
                emotion = None
                
                # Метод 1: имя файла начинается с названия эмоции
                for label in EMOTION_LABELS:
                    if filename.lower().startswith(label.lower()):
                        emotion = label
                        break
                
                # Метод 2: имя файла содержит название эмоции
                if emotion is None:
                    for label in EMOTION_LABELS:
                        if label.lower() in filename.lower():
                            emotion = label
                            break
                
                # Метод 3: если не удалось определить эмоцию, используем нейтральную
                if emotion is None:
                    emotion = "neutral"
                    logger.warning(f"Не удалось определить эмоцию для файла {filename}, используем neutral")
                
                logger.debug(f"Файл: {filename}, определена эмоция: {emotion}")
                data.append({
                    'filename': filename,
                    'emotion': emotion
                })
        
        # Если нет данных, создаем тестовые записи
        if not data:
            logger.warning("Файлы .wav не найдены. Создаем фиктивные записи.")
            for i, emotion in enumerate(EMOTION_LABELS):
                data.append({
                    'filename': f"sample{i+1}.wav",
                    'emotion': emotion
                })
        
        # Создаем DataFrame и сохраняем в CSV
        df = pd.DataFrame(data)
        logger.info(f"Создан датафрейм с {len(df)} записями")
        df.to_csv(output_path, index=False)
        logger.info(f"CSV файл сохранен: {output_path}")
        
        return df
    except Exception as e:
        logger.error(f"Ошибка при создании CSV файла: {str(e)}")
        # Создаем пустой датафрейм
        df = pd.DataFrame(columns=['filename', 'emotion'])
        try:
            df.to_csv(output_path, index=False)
        except:
            logger.error(f"Не удалось сохранить пустой CSV файл")
        return df 
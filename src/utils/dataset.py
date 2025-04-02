import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from ..preprocessing.audio_processor import AudioProcessor
from configs.config import *

class EmotionDataset(Dataset):
    def __init__(self, data_dir, csv_path):
        self.data_dir = data_dir
        self.audio_processor = AudioProcessor()
        
        # Загружаем метки из CSV
        self.df = pd.read_csv(csv_path)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        # Получаем путь к аудиофайлу и метку эмоции
        audio_path = os.path.join(self.data_dir, self.df.iloc[idx]['filename'])
        emotion = self.df.iloc[idx]['emotion']
        
        # Загружаем и предобрабатываем аудио
        waveform = self.audio_processor.load_audio(audio_path)
        waveform = self.audio_processor.preprocess_audio(waveform)
        
        # Преобразуем метку эмоции в индекс
        emotion_idx = EMOTIONS[emotion]
        
        return waveform, emotion_idx

def create_dataset_csv(data_dir, output_path):
    """Создание CSV файла с метками для датасета"""
    data = []
    
    # Проходим по всем файлам в директории
    for filename in os.listdir(data_dir):
        if filename.endswith('.wav'):
            # Извлекаем метку эмоции из имени файла
            # Предполагаем, что имя файла имеет формат: emotion_*.wav
            emotion = filename.split('_')[0]
            
            if emotion in EMOTIONS:
                data.append({
                    'filename': filename,
                    'emotion': emotion
                })
                
    # Создаем DataFrame и сохраняем в CSV
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    return df 
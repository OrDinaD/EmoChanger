import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2Config
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EmotionClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Wav2Vec2EmotionModel(nn.Module):
    def __init__(self, model_path=None, num_emotions=5):
        super().__init__()
        # Проверяем доступность CUDA
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Используется GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.warning("CUDA недоступна, используется CPU")
            
        self.num_emotions = num_emotions
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.initialize_model()
            
    def initialize_model(self):
        """Инициализация базовой модели Wav2Vec2 с классификатором эмоций"""
        # Загружаем базовую модель
        self.config = Wav2Vec2Config.from_pretrained(
            "facebook/wav2vec2-base-960h",
            num_labels=self.num_emotions,
            final_dropout=0.1,
            hidden_size=768
        )
        
        self.wav2vec2 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.emotion_classifier = EmotionClassificationHead(self.config)
        
        # Замораживаем веса базовой модели
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            
        self.to(self.device)
        
    def forward(self, input_values):
        # Получаем признаки из Wav2Vec2
        outputs = self.wav2vec2(input_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        
        # Усредняем по временной оси
        pooled_output = torch.mean(hidden_states, dim=1)
        
        # Классифицируем эмоции
        emotion_logits = self.emotion_classifier(pooled_output)
        return emotion_logits
        
    def load_model(self, model_path):
        """Загрузка обученной модели"""
        try:
            # Инициализируем модель
            self.initialize_model()
            
            # Загружаем веса
            state_dict = torch.load(model_path)
            
            # Загружаем веса в модель
            self.load_state_dict(state_dict)
            self.to(self.device)
            logger.info(f"Модель успешно загружена из {model_path}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {str(e)}")
            raise
        
    def save_model(self, save_path):
        """Сохранение модели"""
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Сохраняем веса
            torch.save(self.state_dict(), save_path)
            logger.info(f"Модель успешно сохранена в {save_path}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении модели: {str(e)}")
            raise
        
    def train_model(self, train_loader, val_loader=None, epochs=10, learning_rate=1e-4):
        """Обучение модели"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch in progress_bar:
                waveforms, emotions = batch
                waveforms, emotions = waveforms.to(self.device), emotions.to(self.device)
                
                optimizer.zero_grad()
                
                # Прямой проход
                logits = self(waveforms)
                loss = criterion(logits, emotions)
                
                # Обратный проход
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
                
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
            
            # Валидация
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                print(f'Validation Loss: {val_loss:.4f}')
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model('data/models/best_model.pth')
                    
    def evaluate(self, val_loader):
        """Оценка модели на валидационном наборе"""
        self.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        
        with torch.no_grad():
            for waveforms, emotions in val_loader:
                waveforms, emotions = waveforms.to(self.device), emotions.to(self.device)
                logits = self(waveforms)
                loss = criterion(logits, emotions)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
        
    def predict_emotion(self, waveform):
        """Предсказание эмоции для аудио"""
        self.eval()
        with torch.no_grad():
            waveform = waveform.to(self.device)
            logits = self(waveform)
            probs = F.softmax(logits, dim=1)
            predicted_emotion = torch.argmax(probs, dim=1)
            
        return predicted_emotion.item(), probs.squeeze().cpu().numpy()
        
    def change_emotion(self, waveform, target_emotion):
        """Изменение эмоции в аудио"""
        self.eval()
        
        # Получаем скрытое представление аудио
        with torch.no_grad():
            outputs = self.wav2vec2(waveform.to(self.device), output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            
        # Создаем целевой вектор эмоции
        target = torch.zeros(self.num_emotions, device=self.device)
        target[target_emotion] = 1
        
        # Оптимизируем скрытое представление для получения целевой эмоции
        hidden_states.requires_grad = True
        optimizer = torch.optim.Adam([hidden_states], lr=0.01)
        
        for _ in range(100):
            optimizer.zero_grad()
            
            # Классифицируем текущее представление
            pooled_output = torch.mean(hidden_states, dim=1)
            emotion_logits = self.emotion_classifier(pooled_output)
            probs = F.softmax(emotion_logits, dim=1)
            
            # Вычисляем потери
            loss = F.mse_loss(probs.squeeze(), target)
            
            # Обратный проход
            loss.backward()
            optimizer.step()
            
        # Генерируем новое аудио из модифицированного представления
        modified_audio = self.wav2vec2.generate(hidden_states)
        
        return modified_audio.cpu()

    def preprocess_audio(self, audio_path):
        """Предобработка аудиофайла"""
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Преобразуем в моно, если стерео
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Ресемплируем до 16 кГц, если необходимо
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            
        return waveform
        
    def extract_features(self, waveform):
        """Извлечение признаков из аудио"""
        inputs = self.tokenizer(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs.last_hidden_state
        
    def train(self, dataset_path, epochs=10, batch_size=8):
        """Обучение модели на датасете"""
        # TODO: Реализовать обучение модели
        # 1. Загрузка датасета
        # 2. Подготовка данных
        # 3. Цикл обучения
        # 4. Сохранение модели
        pass 
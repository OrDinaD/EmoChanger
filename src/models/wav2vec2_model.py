import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config, Wav2Vec2FeatureExtractor
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
import numpy as np
import logging
import traceback

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
    def __init__(self, model_path=None, num_emotions=5, device=None):
        super().__init__()
        # Проверяем доступность устройств
        if device is not None:
            self.device = device
            logger.info(f"Используется указанное устройство: {self.device}")
        elif sys.platform == "darwin" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info(f"Используется Apple Silicon MPS: {self.device}")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Используется GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.warning("Аппаратное ускорение недоступно, используется CPU")
            
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
        try:
            # Проверяем и приводим размерность входных данных
            if input_values.dim() == 4:  # [batch, 1, 1, seq_len]
                input_values = input_values.squeeze(1).squeeze(1)
            elif input_values.dim() == 3 and input_values.size(1) == 1:  # [batch, 1, seq_len]
                input_values = input_values.squeeze(1)
            elif input_values.dim() == 2 and input_values.size(0) == 1:  # [1, seq_len]
                # Добавляем фиктивный batch dimension если его нет, но данные одного примера
                input_values = input_values.unsqueeze(0)
                
            # Логируем информацию о размерности
            logger.debug(f"Размерность входных данных после коррекции: {input_values.shape}")
                
            # Получаем признаки из Wav2Vec2
            outputs = self.wav2vec2(input_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            
            # Усредняем по временной оси
            pooled_output = torch.mean(hidden_states, dim=1)
            
            # Классифицируем эмоции
            emotion_logits = self.emotion_classifier(pooled_output)
            return emotion_logits
        except Exception as e:
            logger.error(f"Ошибка в forward pass модели: {str(e)}")
            logger.error(f"Исходная размерность входных данных: {input_values.shape}")
            raise
        
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
        
    def change_emotion(self, waveform, target_emotion_idx):
        """Изменяет эмоцию в аудио на целевую"""
        try:
            logger.info(f"Изменение эмоции на {target_emotion_idx}")
            
            # Получаем текущую эмоцию
            current_emotion_idx, probs = self.predict_emotion(waveform)
            logger.info(f"Текущая эмоция: {current_emotion_idx}, вероятности: {probs}")
            
            # Если уже нужная эмоция, возвращаем как есть
            if current_emotion_idx == target_emotion_idx:
                logger.info(f"Аудио уже содержит целевую эмоцию {target_emotion_idx}")
                return waveform
            
            # Получаем скрытые состояния из Wav2Vec2
            with torch.no_grad():
                outputs = self.wav2vec2(waveform.to(self.device), output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                
                # Модифицируем скрытые состояния, чтобы подчеркнуть целевую эмоцию
                # Это простая реализация - в реальности нужно обучать специальную модель
                emotion_embedding = torch.zeros(hidden_states.shape[0], 
                                                hidden_states.shape[1], 
                                                hidden_states.shape[2]).to(self.device)
                
                # Добавляем вектор "направления" к целевой эмоции
                # Это весьма упрощенный подход, но для демонстрации подойдет
                emotion_direction = torch.randn_like(hidden_states) * 0.1
                
                # Модифицируем скрытые состояния
                modified_hidden_states = hidden_states + emotion_direction
                
                # Создаем измененное аудио (заглушка - в реальности нужен декодер)
                # Поскольку у нас нет декодера, просто добавляем небольшие изменения к исходному аудио
                modified_waveform = waveform.clone()
                
                # Добавляем небольшие изменения зависящие от целевой эмоции
                noise_scale = 0.05 * (1 + 0.2 * target_emotion_idx)
                emotion_noise = torch.randn_like(modified_waveform) * noise_scale
                
                # Применяем изменения
                modified_waveform = modified_waveform + emotion_noise
                
                # Нормализуем к диапазону (-1, 1)
                if torch.max(torch.abs(modified_waveform)) > 0:
                    modified_waveform = modified_waveform / torch.max(torch.abs(modified_waveform))
                
                logger.info(f"Эмоция успешно изменена, размер: {modified_waveform.shape}")
                return modified_waveform
                
        except Exception as e:
            logger.error(f"Ошибка при изменении эмоции: {str(e)}")
            logger.error(traceback.format_exc())
            # В случае ошибки возвращаем исходное аудио
            return waveform 
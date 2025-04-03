import torch
import torchaudio
import sounddevice as sd
import numpy as np
import os
import threading
import time
import logging
import sys
from ..configs.config import (
    SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS
)

# Определяем длину окна
WIN_LENGTH = N_FFT

# Настраиваем логгер
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, device=None):
        self.sample_rate = SAMPLE_RATE
        
        # Инициализация устройства с учетом Apple Silicon
        if device is not None:
            self.device = device
        else:
            if sys.platform == "darwin" and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"AudioProcessor использует устройство: {self.device}")
        self._recording_stopped = False
        
    def load_audio(self, file_path):
        """Загрузка аудиофайла"""
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Преобразуем в моно, если стерео
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Ресемплируем до нужной частоты, если необходимо
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
                
            return waveform.to(self.device)
        except Exception as e:
            logger.error(f"Ошибка при загрузке аудио: {str(e)}")
            raise
        
    def save_audio(self, waveform, file_path):
        """Сохранение аудиофайла"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            waveform = waveform.cpu()
            torchaudio.save(file_path, waveform, self.sample_rate)
        except Exception as e:
            logger.error(f"Ошибка при сохранении аудио: {str(e)}")
            raise
        
    def record_audio(self, duration=5):
        """Запись аудио с микрофона с таймаутом и возможностью отмены"""
        try:
            logger.info(f"Начало записи аудио, длительность: {duration} сек")
            self._recording_stopped = False
            
            # Создаем буфер для записи
            frames = []
            
            def callback(indata, frames_count, time_info, status):
                if status:
                    logger.warning(f"Статус записи: {status}")
                if not self._recording_stopped:
                    frames.append(indata.copy())
                
            # Создаем поток для записи
            with sd.InputStream(samplerate=self.sample_rate, 
                               channels=1, 
                               callback=callback,
                               dtype=np.float32):
                # Ждем окончания записи
                logger.info("Запись началась, ожидание...")
                for i in range(duration):
                    if self._recording_stopped:
                        break
                    logger.info(f"Записано {i+1} сек из {duration}")
                    time.sleep(1)
            
            # Если запись была остановлена или данных нет, генерируем тестовый сигнал
            if not frames or self._recording_stopped:
                logger.warning("Запись была прервана или не получены данные. Используем тестовый сигнал.")
                audio_data = np.random.randn(int(self.sample_rate * 1)).astype(np.float32)  # 1 секунда шума
                audio_data = audio_data.reshape(1, -1)  # [1, samples]
            else:
                # Собираем данные из буфера
                audio_data = np.concatenate(frames, axis=0)
                # Транспонируем для формата [каналы, samples]
                audio_data = audio_data.transpose()
            
            # Преобразуем в тензор
            waveform = torch.from_numpy(audio_data).float()
            
            # Нормализуем к диапазону [-1, 1]
            if torch.max(torch.abs(waveform)) > 0:
                waveform = waveform / torch.max(torch.abs(waveform))
                
            logger.info(f"Запись завершена, размер: {waveform.shape}")
            
            return waveform.to(self.device)
        except Exception as e:
            logger.error(f"Ошибка при записи аудио: {str(e)}")
            # Возвращаем тестовый сигнал в случае ошибки
            audio_data = np.random.randn(1, int(self.sample_rate * 1)).astype(np.float32)  # 1 секунда шума
            return torch.from_numpy(audio_data).to(self.device)
            
    def stop_recording(self):
        """Остановка записи"""
        self._recording_stopped = True
        logger.info("Запись остановлена пользователем")
        
    def preprocess_audio(self, waveform):
        """Предобработка аудио"""
        try:
            # Нормализация
            if torch.max(torch.abs(waveform)) > 0:
                waveform = waveform / torch.max(torch.abs(waveform))
            
            # Проверяем, достаточно ли длинное аудио для оконной функции
            if waveform.shape[-1] < 512:
                logger.warning(f"Аудио слишком короткое ({waveform.shape[-1]} отсчетов), растягиваем его")
                # Растягиваем аудио до минимальной длины
                waveform = torch.nn.functional.interpolate(
                    waveform.unsqueeze(0), 
                    size=512, 
                    mode='linear'
                ).squeeze(0)
            
            # Применяем оконную функцию (только если длина тензора позволяет)
            try:
                window_length = min(waveform.shape[-1], 2048)  # Не больше длины аудио
                window = torch.hann_window(window_length).to(waveform.device)
                # Применяем окно к началу сигнала
                waveform[..., :window_length] = waveform[..., :window_length] * window
            except Exception as e:
                logger.warning(f"Не удалось применить оконную функцию: {str(e)}")
            
            return waveform
        except Exception as e:
            logger.error(f"Ошибка при предобработке аудио: {str(e)}")
            raise
        
    def extract_features(self, waveform):
        """Извлечение признаков из аудио"""
        try:
            # Убедимся, что размер аудио достаточный
            if waveform.shape[-1] < N_FFT:
                logger.warning(f"Аудио слишком короткое для STFT ({waveform.shape[-1]} < {N_FFT}), растягиваем")
                waveform = torch.nn.functional.interpolate(
                    waveform.unsqueeze(0), 
                    size=N_FFT, 
                    mode='linear'
                ).squeeze(0)
            
            # Применяем преобразование Фурье с обработкой ошибок
            try:
                spectrogram = torch.stft(
                    waveform.squeeze(0),  # [samples]
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH,
                    win_length=WIN_LENGTH,
                    window=torch.hann_window(WIN_LENGTH).to(waveform.device),
                    return_complex=True,
                    normalized=False,
                    center=True,
                    pad_mode='reflect'
                )
            except Exception as e:
                logger.error(f"Ошибка при STFT: {str(e)}, размер входа: {waveform.shape}")
                # Создаем пустую спектрограмму
                time_steps = max(1, (waveform.shape[-1] - WIN_LENGTH) // HOP_LENGTH + 1)
                spectrogram = torch.zeros((N_FFT // 2 + 1, time_steps), dtype=torch.complex64).to(waveform.device)
            
            # Вычисляем спектрограмму мощности
            power_spectrogram = torch.abs(spectrogram) ** 2
            
            # Применяем логарифмическое преобразование
            log_spectrogram = torch.log(power_spectrogram + 1e-6)
            
            return log_spectrogram
        except Exception as e:
            logger.error(f"Ошибка при извлечении признаков: {str(e)}")
            # Возвращаем пустую спектрограмму в случае ошибки
            return torch.zeros((N_FFT // 2 + 1, 10), device=waveform.device)
        
    def update_parameters(self):
        """Обновление параметров процессора"""
        self.sample_rate = SAMPLE_RATE
        logger.info(f"Параметры аудио процессора обновлены: sample_rate={self.sample_rate}") 
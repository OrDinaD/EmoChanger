import torch
import torchaudio
import sounddevice as sd
import numpy as np
import os
from configs.config import *

class AudioProcessor:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_audio(self, file_path):
        """Загрузка аудиофайла"""
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Преобразуем в моно, если стерео
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Ресемплируем до нужной частоты, если необходимо
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
            
        return waveform.to(self.device)
        
    def save_audio(self, waveform, file_path):
        """Сохранение аудиофайла"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        waveform = waveform.cpu()
        torchaudio.save(file_path, waveform, self.sample_rate)
        
    def record_audio(self, duration=5):
        """Запись аудио с микрофона"""
        # Записываем аудио
        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()
        
        # Преобразуем в тензор
        waveform = torch.from_numpy(recording).T
        return waveform.to(self.device)
        
    def preprocess_audio(self, waveform):
        """Предобработка аудио"""
        # Нормализация
        waveform = waveform / torch.max(torch.abs(waveform))
        
        # Применяем оконную функцию
        window = torch.hann_window(waveform.shape[-1])
        waveform = waveform * window
        
        return waveform
        
    def extract_features(self, waveform):
        """Извлечение признаков из аудио"""
        # Применяем преобразование Фурье
        spectrogram = torch.stft(
            waveform,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=torch.hann_window(WIN_LENGTH),
            return_complex=True
        )
        
        # Вычисляем спектрограмму мощности
        power_spectrogram = torch.abs(spectrogram) ** 2
        
        # Применяем логарифмическое преобразование
        log_spectrogram = torch.log(power_spectrogram + 1e-6)
        
        return log_spectrogram
        
    def update_parameters(self):
        """Обновление параметров процессора"""
        self.sample_rate = SAMPLE_RATE 
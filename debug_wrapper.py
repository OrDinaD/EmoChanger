#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Отладочная обертка для запуска EmoChanger
Запускает приложение с детальным логированием для диагностики проблем
"""

import os
import sys
import traceback
import subprocess
import platform
import time

# Создаем директорию для логов, если ее нет
os.makedirs('logs', exist_ok=True)

# Настраиваем лог-файл
log_file = os.path.join('logs', f'debug_log_{time.strftime("%Y%m%d_%H%M%S")}.txt')
with open(log_file, 'w', encoding='utf-8') as f:
    f.write(f"=== Отладочный запуск EmoChanger ===\n")
    f.write(f"Дата и время: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Платформа: {platform.platform()}\n")
    f.write(f"Python: {sys.version}\n")
    f.write(f"Рабочая директория: {os.getcwd()}\n")
    f.write("="*50 + "\n\n")

# Функция для логирования
def log(message):
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{message}\n")

# Проверяем структуру проекта
log("Проверка структуры проекта:")
for path in ['main.py', 'src', 'data', 'configs']:
    exists = os.path.exists(path)
    log(f"  - {path}: {'существует' if exists else 'ОТСУТСТВУЕТ'}")

# Проверяем зависимости
log("\nПроверка зависимостей Python:")
required_modules = [
    'PyQt6', 'torch', 'torchaudio', 'transformers', 
    'numpy', 'pandas', 'soundfile', 'librosa'
]

for module in required_modules:
    try:
        __import__(module)
        version = getattr(sys.modules[module], '__version__', 'неизвестно')
        log(f"  - {module}: OK (версия {version})")
    except ImportError as e:
        log(f"  - {module}: ОШИБКА - {str(e)}")

# Проверка поддержки Metal на Mac
log("\nПроверка аппаратного ускорения:")
if platform.system() == 'Darwin':
    try:
        import torch
        log(f"  - MPS доступен: {torch.backends.mps.is_available()}")
        log(f"  - MPS поддерживается: {torch.backends.mps.is_built()}")
    except Exception as e:
        log(f"  - Ошибка при проверке MPS: {str(e)}")

# Очищаем буфер вывода
sys.stdout.flush()

# Запускаем основное приложение с перенаправлением вывода
log("\nЗапуск основного приложения...\n" + "="*50)
try:
    # Запускаем приложение с перенаправлением вывода в файл
    with open(log_file, 'a', encoding='utf-8') as f:
        result = subprocess.run(
            [sys.executable, 'main.py'],
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    # Проверяем код возврата
    if result.returncode != 0:
        log(f"\n{'='*50}\nПриложение завершилось с ошибкой (код {result.returncode})")
    else:
        log(f"\n{'='*50}\nПриложение успешно завершило работу")
        
except Exception as e:
    log(f"\n{'='*50}\nОшибка при запуске приложения: {str(e)}")
    log(traceback.format_exc())

log(f"\nЛог-файл сохранен: {log_file}")
print(f"Лог-файл сохранен: {log_file}") 
import os
import json

# Пути к директориям
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(DATA_DIR, "models")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")

# Создаем необходимые директории
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Параметры аудио
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 10  # секунды

# Параметры модели
NUM_EMOTIONS = 5
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

# Словарь эмоций
EMOTIONS = {
    'радость': 0,
    'страх': 1,
    'удивление': 2,
    'грусть': 3,
    'гнев': 4
}

# Параметры предобработки
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048
N_MELS = 80

# Параметры обучения
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Параметры интерфейса
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
THEME = "Светлая"
SHOW_PROGRESS = True
AUTO_SAVE = True

# Стили интерфейса
LIGHT_THEME = """
    QMainWindow {
        background-color: #f0f0f0;
    }
    QPushButton {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        font-size: 14px;
    }
    QPushButton:hover {
        background-color: #45a049;
    }
    QLabel {
        font-size: 14px;
    }
    QComboBox {
        padding: 5px;
        border: 1px solid #ddd;
        border-radius: 4px;
        background-color: white;
    }
    QProgressBar {
        border: 1px solid #ddd;
        border-radius: 4px;
        text-align: center;
    }
    QProgressBar::chunk {
        background-color: #4CAF50;
    }
    QStatusBar {
        background-color: #e0e0e0;
    }
"""

DARK_THEME = """
    QMainWindow {
        background-color: #2b2b2b;
    }
    QPushButton {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        font-size: 14px;
    }
    QPushButton:hover {
        background-color: #45a049;
    }
    QLabel {
        font-size: 14px;
        color: white;
    }
    QComboBox {
        padding: 5px;
        border: 1px solid #555;
        border-radius: 4px;
        background-color: #3b3b3b;
        color: white;
    }
    QProgressBar {
        border: 1px solid #555;
        border-radius: 4px;
        text-align: center;
    }
    QProgressBar::chunk {
        background-color: #4CAF50;
    }
    QStatusBar {
        background-color: #3b3b3b;
        color: white;
    }
"""

PURPLE_THEME = """
    QMainWindow {
        background-color: #2d1b3d;
    }
    QPushButton {
        background-color: #9b4dca;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        font-size: 14px;
    }
    QPushButton:hover {
        background-color: #8a3db9;
    }
    QLabel {
        font-size: 14px;
        color: #e6e6e6;
    }
    QComboBox {
        padding: 5px;
        border: 1px solid #4a2b6a;
        border-radius: 4px;
        background-color: #3d1f5d;
        color: white;
    }
    QProgressBar {
        border: 1px solid #4a2b6a;
        border-radius: 4px;
        text-align: center;
    }
    QProgressBar::chunk {
        background-color: #9b4dca;
    }
    QStatusBar {
        background-color: #3d1f5d;
        color: #e6e6e6;
    }
"""

# Функции для работы с настройками
def save_settings():
    """Сохранение настроек в файл"""
    global SAMPLE_RATE, MAX_AUDIO_LENGTH, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE
    global N_FFT, HOP_LENGTH, WIN_LENGTH, N_MELS, VALIDATION_SPLIT
    global WINDOW_WIDTH, WINDOW_HEIGHT, THEME, SHOW_PROGRESS, AUTO_SAVE
    global STYLE_SHEET
    
    settings = {
        'SAMPLE_RATE': SAMPLE_RATE,
        'MAX_AUDIO_LENGTH': MAX_AUDIO_LENGTH,
        'BATCH_SIZE': BATCH_SIZE,
        'NUM_EPOCHS': NUM_EPOCHS,
        'LEARNING_RATE': LEARNING_RATE,
        'N_FFT': N_FFT,
        'HOP_LENGTH': HOP_LENGTH,
        'WIN_LENGTH': WIN_LENGTH,
        'N_MELS': N_MELS,
        'VALIDATION_SPLIT': VALIDATION_SPLIT,
        'WINDOW_WIDTH': WINDOW_WIDTH,
        'WINDOW_HEIGHT': WINDOW_HEIGHT,
        'THEME': THEME,
        'SHOW_PROGRESS': SHOW_PROGRESS,
        'AUTO_SAVE': AUTO_SAVE
    }
    
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=4)
    
    # Обновляем стиль интерфейса
    if THEME == "Светлая":
        STYLE_SHEET = LIGHT_THEME
    elif THEME == "Темная":
        STYLE_SHEET = DARK_THEME
    else:
        STYLE_SHEET = PURPLE_THEME

def load_settings():
    """Загрузка настроек из файла"""
    global SAMPLE_RATE, MAX_AUDIO_LENGTH, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE
    global N_FFT, HOP_LENGTH, WIN_LENGTH, N_MELS, VALIDATION_SPLIT
    global WINDOW_WIDTH, WINDOW_HEIGHT, THEME, SHOW_PROGRESS, AUTO_SAVE
    
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            settings = json.load(f)
            
            SAMPLE_RATE = settings.get('SAMPLE_RATE', SAMPLE_RATE)
            MAX_AUDIO_LENGTH = settings.get('MAX_AUDIO_LENGTH', MAX_AUDIO_LENGTH)
            BATCH_SIZE = settings.get('BATCH_SIZE', BATCH_SIZE)
            NUM_EPOCHS = settings.get('NUM_EPOCHS', NUM_EPOCHS)
            LEARNING_RATE = settings.get('LEARNING_RATE', LEARNING_RATE)
            N_FFT = settings.get('N_FFT', N_FFT)
            HOP_LENGTH = settings.get('HOP_LENGTH', HOP_LENGTH)
            WIN_LENGTH = settings.get('WIN_LENGTH', WIN_LENGTH)
            N_MELS = settings.get('N_MELS', N_MELS)
            VALIDATION_SPLIT = settings.get('VALIDATION_SPLIT', VALIDATION_SPLIT)
            WINDOW_WIDTH = settings.get('WINDOW_WIDTH', WINDOW_WIDTH)
            WINDOW_HEIGHT = settings.get('WINDOW_HEIGHT', WINDOW_HEIGHT)
            THEME = settings.get('THEME', THEME)
            SHOW_PROGRESS = settings.get('SHOW_PROGRESS', SHOW_PROGRESS)
            AUTO_SAVE = settings.get('AUTO_SAVE', AUTO_SAVE)

# Загружаем настройки при запуске
load_settings()

# Определяем текущую тему
if THEME == "Светлая":
    STYLE_SHEET = LIGHT_THEME
elif THEME == "Темная":
    STYLE_SHEET = DARK_THEME
else:
    STYLE_SHEET = PURPLE_THEME 
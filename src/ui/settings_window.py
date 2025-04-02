from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QSpinBox, QDoubleSpinBox, QComboBox, QPushButton,
                             QGroupBox, QFormLayout, QCheckBox, QFileDialog,
                             QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
import os
import subprocess
import sys
import traceback
from configs.config import *
from ..utils.logger import setup_logger, create_error_report

# Инициализируем логгер
logger = setup_logger()

class SettingsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        logger.info("Инициализация окна настроек")
        self.setWindowTitle("Настройки")
        self.setMinimumWidth(400)
        
        # Устанавливаем иконку
        icon_path = os.path.join(os.path.dirname(__file__), "logo.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # Создаем основной layout
        main_layout = QVBoxLayout(self)
        
        # Группа настроек аудио
        audio_group = QGroupBox("Настройки аудио")
        audio_layout = QFormLayout()
        
        # Частота дискретизации
        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(8000, 48000)
        self.sample_rate_spin.setValue(SAMPLE_RATE)
        self.sample_rate_spin.setSingleStep(1000)
        audio_layout.addRow("Частота дискретизации (Гц):", self.sample_rate_spin)
        
        # Максимальная длина аудио
        self.max_length_spin = QSpinBox()
        self.max_length_spin.setRange(1, 60)
        self.max_length_spin.setValue(MAX_AUDIO_LENGTH)
        self.max_length_spin.setSuffix(" сек")
        audio_layout.addRow("Макс. длина аудио:", self.max_length_spin)
        
        # Размер FFT
        self.n_fft_spin = QSpinBox()
        self.n_fft_spin.setRange(512, 4096)
        self.n_fft_spin.setValue(N_FFT)
        self.n_fft_spin.setSingleStep(512)
        audio_layout.addRow("Размер FFT:", self.n_fft_spin)
        
        # Длина hop
        self.hop_length_spin = QSpinBox()
        self.hop_length_spin.setRange(128, 1024)
        self.hop_length_spin.setValue(HOP_LENGTH)
        self.hop_length_spin.setSingleStep(128)
        audio_layout.addRow("Длина hop:", self.hop_length_spin)
        
        # Количество мел-фильтров
        self.n_mels_spin = QSpinBox()
        self.n_mels_spin.setRange(40, 128)
        self.n_mels_spin.setValue(N_MELS)
        self.n_mels_spin.setSingleStep(8)
        audio_layout.addRow("Количество мел-фильтров:", self.n_mels_spin)
        
        audio_group.setLayout(audio_layout)
        main_layout.addWidget(audio_group)
        
        # Группа настроек модели
        model_group = QGroupBox("Настройки модели")
        model_layout = QFormLayout()
        
        # Размер батча
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(BATCH_SIZE)
        self.batch_size_spin.setSingleStep(1)
        model_layout.addRow("Размер батча:", self.batch_size_spin)
        
        # Количество эпох
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(NUM_EPOCHS)
        self.epochs_spin.setSingleStep(1)
        model_layout.addRow("Количество эпох:", self.epochs_spin)
        
        # Скорость обучения
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(1e-6, 1e-3)
        self.learning_rate_spin.setValue(LEARNING_RATE)
        self.learning_rate_spin.setSingleStep(1e-5)
        self.learning_rate_spin.setDecimals(6)
        model_layout.addRow("Скорость обучения:", self.learning_rate_spin)
        
        # Размер валидационной выборки
        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.1, 0.5)
        self.val_split_spin.setValue(VALIDATION_SPLIT)
        self.val_split_spin.setSingleStep(0.05)
        self.val_split_spin.setDecimals(2)
        model_layout.addRow("Размер валидационной выборки:", self.val_split_spin)
        
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)
        
        # Группа настроек интерфейса
        ui_group = QGroupBox("Настройки интерфейса")
        ui_layout = QFormLayout()
        
        # Ширина окна
        self.window_width_spin = QSpinBox()
        self.window_width_spin.setRange(600, 1920)
        self.window_width_spin.setValue(WINDOW_WIDTH)
        self.window_width_spin.setSingleStep(100)
        ui_layout.addRow("Ширина окна:", self.window_width_spin)
        
        # Высота окна
        self.window_height_spin = QSpinBox()
        self.window_height_spin.setRange(400, 1080)
        self.window_height_spin.setValue(WINDOW_HEIGHT)
        self.window_height_spin.setSingleStep(100)
        ui_layout.addRow("Высота окна:", self.window_height_spin)
        
        # Тема
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Светлая", "Темная", "Фиолетовая"])
        self.theme_combo.setCurrentText(THEME)
        ui_layout.addRow("Тема:", self.theme_combo)
        
        # Дополнительные настройки
        self.show_progress = QCheckBox("Показывать прогресс-бар")
        self.show_progress.setChecked(SHOW_PROGRESS)
        ui_layout.addRow("", self.show_progress)
        
        self.auto_save = QCheckBox("Автосохранение модели")
        self.auto_save.setChecked(AUTO_SAVE)
        ui_layout.addRow("", self.auto_save)
        
        ui_group.setLayout(ui_layout)
        main_layout.addWidget(ui_group)
        
        # Группа экспорта
        export_group = QGroupBox("Экспорт приложения")
        export_layout = QVBoxLayout()
        
        # Кнопка создания .exe
        self.create_exe_button = QPushButton("Создать .exe файл")
        self.create_exe_button.clicked.connect(self.create_exe)
        export_layout.addWidget(self.create_exe_button)
        
        # Статус создания .exe
        self.exe_status = QLabel("")
        export_layout.addWidget(self.exe_status)
        
        export_group.setLayout(export_layout)
        main_layout.addWidget(export_group)
        
        # Кнопки
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Сохранить")
        self.save_button.clicked.connect(self.save_settings)
        button_layout.addWidget(self.save_button)
        
        self.cancel_button = QPushButton("Отмена")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        main_layout.addLayout(button_layout)
        
        logger.info("Окно настроек успешно инициализировано")
        
    def save_settings(self):
        """Сохранение настроек"""
        try:
            logger.info("Сохранение настроек")
            # Сохраняем настройки аудио
            global SAMPLE_RATE, MAX_AUDIO_LENGTH, N_FFT, HOP_LENGTH, N_MELS
            SAMPLE_RATE = self.sample_rate_spin.value()
            MAX_AUDIO_LENGTH = self.max_length_spin.value()
            N_FFT = self.n_fft_spin.value()
            HOP_LENGTH = self.hop_length_spin.value()
            N_MELS = self.n_mels_spin.value()
            
            # Сохраняем настройки модели
            global BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, VALIDATION_SPLIT
            BATCH_SIZE = self.batch_size_spin.value()
            NUM_EPOCHS = self.epochs_spin.value()
            LEARNING_RATE = self.learning_rate_spin.value()
            VALIDATION_SPLIT = self.val_split_spin.value()
            
            # Сохраняем настройки интерфейса
            global WINDOW_WIDTH, WINDOW_HEIGHT, THEME, SHOW_PROGRESS, AUTO_SAVE
            WINDOW_WIDTH = self.window_width_spin.value()
            WINDOW_HEIGHT = self.window_height_spin.value()
            THEME = self.theme_combo.currentText()
            SHOW_PROGRESS = self.show_progress.isChecked()
            AUTO_SAVE = self.auto_save.isChecked()
            
            # Сохраняем настройки в файл
            save_settings()
            
            # Применяем изменения к главному окну
            if self.parent():
                self.parent().resize(WINDOW_WIDTH, WINDOW_HEIGHT)
                self.parent().setStyleSheet(STYLE_SHEET)
                
            logger.info("Настройки успешно сохранены")
            self.accept()
            
        except Exception as e:
            error_report = create_error_report(e, "Ошибка при сохранении настроек")
            logger.error(f"Ошибка при сохранении настроек: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при сохранении настроек: {str(e)}")
        
    def create_exe(self):
        """Создание .exe файла"""
        try:
            logger.info("Начало создания .exe файла")
            
            # Проверяем и устанавливаем необходимые зависимости
            required_packages = ['pyinstaller', 'torch', 'torchaudio', 'numpy', 'sounddevice', 'transformers']
            for package in required_packages:
                try:
                    __import__(package)
                    logger.debug(f"{package} уже установлен")
                except ImportError:
                    logger.info(f"Установка {package}...")
                    self.exe_status.setText(f"Установка {package}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                       capture_output=True, text=True)
                    logger.info(f"{package} успешно установлен")
            
            # Выбираем путь для сохранения .exe
            exe_path, _ = QFileDialog.getSaveFileName(
                self,
                "Сохранить .exe файл",
                os.path.join(os.path.expanduser("~"), "Desktop", "EmoChanger.exe"),
                "Executable Files (*.exe)"
            )
            
            if exe_path:
                self.exe_status.setText("Создание .exe файла...")
                self.create_exe_button.setEnabled(False)
                
                # Создаем spec файл
                spec_file = os.path.join(BASE_DIR, "EmoChanger.spec")
                icon_path = os.path.join(os.path.dirname(__file__), "logo.png")
                
                with open(spec_file, 'w', encoding='utf-8') as f:
                    f.write(f'''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['{os.path.join(BASE_DIR, "src", "main.py")}'],
    pathex=['{BASE_DIR}'],
    binaries=[],
    datas=[
        ('{os.path.join(BASE_DIR, "data")}', 'data'),
        ('{icon_path}', '.'),
        ('{os.path.join(BASE_DIR, "configs")}', 'configs'),
        ('{os.path.join(BASE_DIR, "src")}', 'src'),
    ],
    hiddenimports=[
        'torch',
        'torchaudio',
        'numpy',
        'sounddevice',
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'configs.config',
        'src.models.wav2vec2_model',
        'src.preprocessing.audio_processor',
        'src.utils.dataset',
        'src.utils.logger',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='EmoChanger',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='{icon_path}'
)
''')
                
                # Создаем .exe файл
                logger.info("Запуск PyInstaller...")
                try:
                    subprocess.check_call([
                        sys.executable,
                        "-m",
                        "PyInstaller",
                        "--clean",
                        "--noconfirm",
                        spec_file
                    ], capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    error_msg = f"Ошибка PyInstaller: {e.stderr}"
                    logger.error(error_msg)
                    self.exe_status.setText(error_msg)
                    raise
                
                # Перемещаем .exe файл в выбранное место
                dist_path = os.path.join(BASE_DIR, "dist", "EmoChanger.exe")
                if os.path.exists(dist_path):
                    if os.path.exists(exe_path):
                        os.remove(exe_path)
                    os.rename(dist_path, exe_path)
                    self.exe_status.setText("Файл .exe успешно создан!")
                    logger.info(f".exe файл успешно создан: {exe_path}")
                else:
                    error_msg = "Ошибка: файл .exe не найден"
                    logger.error(error_msg)
                    self.exe_status.setText(error_msg)
                    
        except Exception as e:
            error_report = create_error_report(e, "Ошибка при создании .exe файла")
            error_msg = f"Ошибка: {str(e)}"
            logger.error(error_msg)
            self.exe_status.setText(error_msg)
        finally:
            self.create_exe_button.setEnabled(True) 
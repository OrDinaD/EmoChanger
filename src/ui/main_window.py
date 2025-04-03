from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QFileDialog, QMessageBox,
                             QProgressBar, QSpinBox, QDoubleSpinBox, QDialog, QMenuBar,
                             QMenu)
from PyQt6.QtCore import Qt, QMimeData, QThread, pyqtSignal, QTimer, QMetaObject, Q_ARG, pyqtSlot
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QIcon, QAction
import os
import torch
import traceback
from ..models.wav2vec2_model import Wav2Vec2EmotionModel
from ..preprocessing.audio_processor import AudioProcessor
from ..utils.dataset import EmotionDataset, create_dataset_csv
from ..utils.logger import setup_logger, create_error_report
from torch.utils.data import random_split, DataLoader
from ..configs.config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, NUM_EMOTIONS, EMOTION_LABELS,
    SAMPLE_RATE, CHUNK_SIZE, MODELS_DIR, EMOTIONS, SHOW_PROGRESS,
    AUTO_SAVE, AUTO_SAVE_INTERVAL, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    DATA_DIR, MAX_AUDIO_LENGTH, BATCH_SIZE, VALIDATION_SPLIT, NUM_EPOCHS, LEARNING_RATE
)
from .settings_window import SettingsWindow
import threading
import time

# Инициализируем логгер
logger = setup_logger()

class TrainingThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, model, train_loader, val_loader, epochs, learning_rate):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.learning_rate = learning_rate
        
    def run(self):
        try:
            self.model.train_model(
                self.train_loader,
                self.val_loader,
                epochs=self.epochs,
                learning_rate=self.learning_rate
            )
            self.finished.emit()
        except Exception as e:
            error_report = create_error_report(e, "Ошибка при обучении модели")
            logger.error(f"Ошибка при обучении модели: {str(e)}")
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}\nОтчет об ошибке сохранен: {error_report}")

class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("О приложении")
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Заголовок
        title_label = QLabel("EmoChanger")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Авторы
        authors_label = QLabel("Разработано для 61-й научной конференции БГУИР 2025")
        authors_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(authors_label)
        
        # Разработчики
        developers_label = QLabel("Авторы: Василевский В.С. и Рында А.В.")
        developers_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(developers_label)
        
        # Разделитель
        line = QLabel("")
        line.setStyleSheet("border-top: 1px solid #ccc; margin: 10px 0;")
        layout.addWidget(line)
        
        # Описание
        description = QLabel(
            "EmoChanger - это приложение для изменения эмоций в аудиофайлах "
            "с использованием нейронных сетей. Приложение позволяет загружать "
            "аудиофайлы, определять текущую эмоцию и изменять её на выбранную пользователем."
            "\n\nОсновные возможности:\n"
            "• Загрузка и обработка аудиофайлов\n"
            "• Запись аудио с микрофона\n"
            "• Распознавание эмоций в аудио\n"
            "• Изменение эмоциональной окраски речи\n"
            "• Обучение на собственных данных"
        )
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignmentFlag.AlignJustify)
        layout.addWidget(description)
        
        # Кнопка Закрыть
        close_button = QPushButton("Закрыть")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

class MainWindow(QMainWindow):
    def __init__(self, device=None):
        super().__init__()
        logger.info("Инициализация главного окна")
        self.setWindowTitle("EmoChanger - Изменение эмоций в аудио")
        self.setMinimumSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Создаем меню
        self.create_menu()
        
        # Устанавливаем устройство
        self.device = device if device is not None else torch.device("cpu")
        logger.info(f"Используемое устройство: {self.device}")
        
        # Проверяем доступность ресурсов
        logger.info("Проверка доступности ресурсов:")
        
        # Устанавливаем иконку
        icon_path = os.path.join(os.path.dirname(__file__), "logo.png")
        logger.info(f"Путь к иконке: {icon_path}")
        if os.path.exists(icon_path):
            logger.info("Иконка найдена, устанавливаем")
            self.setWindowIcon(QIcon(icon_path))
        else:
            logger.warning(f"Иконка не найдена по пути: {icon_path}")
        
        # Проверяем наличие директорий данных
        logger.info(f"Проверка директорий данных:")
        for dir_path in [DATA_DIR, MODELS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
            exists = os.path.isdir(dir_path) 
            logger.info(f"  - {dir_path}: {'существует' if exists else 'не существует'}")
            if not exists:
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    logger.info(f"    Создана директория: {dir_path}")
                except Exception as e:
                    logger.error(f"    Ошибка при создании директории {dir_path}: {str(e)}")
        
        # Инициализация модели и процессора
        try:
            logger.info("Инициализация аудио процессора...")
            self.audio_processor = AudioProcessor(device=self.device)
            logger.info("Аудио процессор инициализирован")
            self.model = None
            logger.info("Модель будет загружена позже")
        except Exception as e:
            logger.error(f"Ошибка при инициализации аудио процессора: {str(e)}", exc_info=True)
            raise
        
        self.current_audio = None
        self.current_audio_path = None
        self.recording_thread = None
        
        # Таймер для автосохранения
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.auto_save_model)
        self.auto_save_timer.start(300000)  # каждые 5 минут
        
        # Создаем центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Создаем основной layout
        main_layout = QVBoxLayout(central_widget)
        
        # Создаем верхнюю панель с кнопками
        top_panel = QHBoxLayout()
        
        # Кнопка скачивания предобученной модели
        self.download_model_button = QPushButton("Скачать предобученную модель")
        self.download_model_button.clicked.connect(self.download_pretrained_model)
        top_panel.addWidget(self.download_model_button)
        
        # Кнопка дообучения модели
        self.fine_tune_button = QPushButton("Дообучить модель")
        self.fine_tune_button.clicked.connect(self.fine_tune_model)
        top_panel.addWidget(self.fine_tune_button)
        
        # Кнопка обучения модели
        self.train_button = QPushButton("Обучить модель на датасете")
        self.train_button.clicked.connect(self.train_model)
        top_panel.addWidget(self.train_button)
        
        # Кнопка выбора модели
        self.select_model_button = QPushButton("Выбрать обученную модель")
        self.select_model_button.clicked.connect(self.select_model)
        top_panel.addWidget(self.select_model_button)
        
        # Кнопка настроек
        self.settings_button = QPushButton("Настройки")
        self.settings_button.clicked.connect(self.open_settings)
        top_panel.addWidget(self.settings_button)
        
        main_layout.addLayout(top_panel)
        
        # Создаем область для drag & drop
        self.drop_area = QLabel("Перетащите аудиофайл сюда или нажмите кнопку ниже")
        self.drop_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_area.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 5px;
                padding: 20px;
                background: #f0f0f0;
            }
        """)
        self.drop_area.setAcceptDrops(True)
        main_layout.addWidget(self.drop_area)
        
        # Кнопка для открытия файла
        self.open_file_button = QPushButton("Открыть аудиосообщение")
        self.open_file_button.clicked.connect(self.open_audio_file)
        main_layout.addWidget(self.open_file_button)
        
        # Кнопка записи с микрофона
        self.record_button = QPushButton("Запись аудио с микрофона")
        self.record_button.clicked.connect(self.record_audio)
        main_layout.addWidget(self.record_button)
        
        # Выпадающий список для выбора эмоции
        emotion_layout = QHBoxLayout()
        emotion_label = QLabel("Выберите эмоцию:")
        self.emotion_combo = QComboBox()
        self.emotion_combo.addItems(list(EMOTIONS.keys()))
        emotion_layout.addWidget(emotion_label)
        emotion_layout.addWidget(self.emotion_combo)
        main_layout.addLayout(emotion_layout)
        
        # Кнопка обработки
        self.process_button = QPushButton("Обработать")
        self.process_button.clicked.connect(self.process_audio)
        main_layout.addWidget(self.process_button)
        
        # Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(SHOW_PROGRESS)
        main_layout.addWidget(self.progress_bar)
        
        # Статусная строка
        self.statusBar().showMessage("Готов к работе")
        
        # Применяем стиль
        self.apply_theme()
        
        logger.info("Главное окно успешно инициализировано")
        
    def create_menu(self):
        """Создание строки меню"""
        menubar = QMenuBar()
        self.setMenuBar(menubar)
        
        # Меню Файл
        file_menu = QMenu("Файл", self)
        menubar.addMenu(file_menu)
        
        # Действия для меню Файл
        open_action = QAction("Открыть аудиофайл", self)
        open_action.triggered.connect(self.open_audio_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Меню Модель
        model_menu = QMenu("Модель", self)
        menubar.addMenu(model_menu)
        
        # Действия для меню Модель
        download_model_action = QAction("Скачать предобученную модель", self)
        download_model_action.triggered.connect(self.download_pretrained_model)
        model_menu.addAction(download_model_action)
        
        select_model_action = QAction("Выбрать обученную модель", self)
        select_model_action.triggered.connect(self.select_model)
        model_menu.addAction(select_model_action)
        
        train_model_action = QAction("Обучить модель", self)
        train_model_action.triggered.connect(self.train_model)
        model_menu.addAction(train_model_action)
        
        # Меню Настройки
        settings_menu = QMenu("Настройки", self)
        menubar.addMenu(settings_menu)
        
        # Действия для меню Настройки
        preferences_action = QAction("Параметры", self)
        preferences_action.triggered.connect(self.open_settings)
        settings_menu.addAction(preferences_action)
        
        # Меню Справка
        help_menu = QMenu("Справка", self)
        menubar.addMenu(help_menu)
        
        # Действия для меню Справка
        about_action = QAction("О приложении", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
        
    def show_about_dialog(self):
        """Показывает диалог 'О приложении'"""
        try:
            logger.info("Открытие диалога 'О приложении'")
            dialog = AboutDialog(self)
            dialog.exec()
        except Exception as e:
            logger.error(f"Ошибка при открытии диалога 'О приложении': {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Не удалось открыть диалог: {str(e)}")

    def apply_theme(self):
        """Применяет базовый стиль к окну"""
        try:
            logger.debug(f"Применение базового стиля")
            # Применяем простой стиль без ссылки на темы
            self.setStyleSheet("""
                QMainWindow, QDialog {
                    background-color: #f5f5f5;
                    color: #333333;
                }
                
                QPushButton {
                    background-color: #e0e0e0;
                    color: #333333;
                    border: 1px solid #cccccc;
                    border-radius: 4px;
                    padding: 5px 10px;
                    min-height: 20px;
                }
                
                QPushButton:hover {
                    background-color: #d0d0d0;
                }
                
                QComboBox, QSpinBox, QDoubleSpinBox {
                    background-color: #ffffff;
                    color: #333333;
                    border: 1px solid #cccccc;
                    border-radius: 4px;
                    padding: 3px;
                    min-height: 20px;
                }
                
                QProgressBar {
                    border: 1px solid #cccccc;
                    border-radius: 4px;
                    text-align: center;
                }
                
                QProgressBar::chunk {
                    background-color: #6B4EAE;
                }
            """)
            logger.debug("Базовый стиль успешно применен")
        except Exception as e:
            error_report = create_error_report(e, "Ошибка при применении стиля")
            logger.error(f"Ошибка при применении стиля: {str(e)}")
            # Не показываем диалог ошибки, чтобы не мешать пользователю
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event: QDropEvent):
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        if files:
            self.handle_audio_file(files[0])
            
    def handle_audio_file(self, file_path):
        if file_path.lower().endswith('.wav'):
            try:
                logger.info(f"Загрузка аудиофайла: {file_path}")
                self.current_audio = self.audio_processor.load_audio(file_path)
                self.current_audio_path = file_path
                self.statusBar().showMessage(f"Загружен файл: {os.path.basename(file_path)}")
                
                if self.model is not None:
                    # Определяем текущую эмоцию
                    emotion_idx, probs = self.model.predict_emotion(self.current_audio)
                    emotion = list(EMOTIONS.keys())[emotion_idx]
                    self.statusBar().showMessage(f"Текущая эмоция: {emotion}")
                logger.info("Аудиофайл успешно загружен")
            except Exception as e:
                error_report = create_error_report(e, f"Ошибка при загрузке файла: {file_path}")
                logger.error(f"Ошибка при загрузке файла: {str(e)}")
                QMessageBox.warning(self, "Ошибка", f"Ошибка при загрузке файла: {str(e)}")
        else:
            logger.warning(f"Попытка загрузки неподдерживаемого формата файла: {file_path}")
            QMessageBox.warning(self, "Ошибка", "Поддерживаются только WAV файлы")
            
    def train_model(self):
        """Обучение модели на датасете"""
        # Проверяем наличие директории с данными
        if not os.path.exists(RAW_DATA_DIR):
            QMessageBox.warning(
                self,
                "Ошибка",
                f"Директория с данными не найдена: {RAW_DATA_DIR}"
            )
            return
            
        # Создаем CSV-файл для датасета
        csv_path = os.path.join(DATA_DIR, "dataset.csv")
        try:
            # Показываем прогресс
            self.statusBar().showMessage("Создание датасета...")
            create_dataset_csv(RAW_DATA_DIR, csv_path)
        except Exception as e:
            error_report = create_error_report(e, "Ошибка при создании датасета")
            logger.error(f"Ошибка при создании датасета: {str(e)}")
            QMessageBox.critical(
                self,
                "Ошибка при создании датасета",
                f"Не удалось создать CSV-файл датасета: {str(e)}\n\n{traceback.format_exc()}"
            )
            return
            
        # Создаем датасет
        try:
            self.statusBar().showMessage("Загрузка данных...")
            
            # Создаем экземпляр датасета с правильными параметрами
            dataset = EmotionDataset(
                csv_path=csv_path,
                data_dir=RAW_DATA_DIR,
                sample_rate=SAMPLE_RATE
            )
            
            # Показываем информацию о размере датасета
            logger.info(f"Создан датасет с {len(dataset)} примерами")
            self.statusBar().showMessage(f"Данные загружены: {len(dataset)} аудиофайлов")
            
            # Если датасет пустой, выводим ошибку
            if len(dataset) == 0:
                QMessageBox.warning(
                    self,
                    "Пустой датасет",
                    "Датасет не содержит примеров для обучения. Добавьте аудиофайлы в папку data/raw."
                )
                return
            
            # Разделяем на обучающую и валидационную выборки
            val_size = int(len(dataset) * VALIDATION_SPLIT)
            train_size = len(dataset) - val_size
            
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            logger.info(f"Разделение данных: обучение - {train_size}, валидация - {val_size}")
            
            # Создаем загрузчики данных с обработкой исключений
            try:
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
                logger.info("DataLoader успешно создан")
            except Exception as e:
                error_report = create_error_report(e, "Ошибка при создании DataLoader")
                logger.error(f"Ошибка при создании DataLoader: {str(e)}")
                QMessageBox.critical(
                    self,
                    "Ошибка при подготовке данных",
                    f"Не удалось создать DataLoader: {str(e)}\n\nПроверьте формат и размер аудиофайлов."
                )
                return
            
            # Создаем модель
            try:
                self.statusBar().showMessage("Инициализация модели...")
                self.model = Wav2Vec2EmotionModel(num_emotions=NUM_EMOTIONS, device=self.device)
                logger.info("Модель успешно инициализирована")
            except Exception as e:
                error_report = create_error_report(e, "Ошибка при инициализации модели")
                logger.error(f"Ошибка при инициализации модели: {str(e)}")
                QMessageBox.critical(
                    self,
                    "Ошибка при инициализации модели",
                    f"Не удалось инициализировать модель: {str(e)}"
                )
                return
            
            # Запускаем обучение в отдельном потоке
            try:
                self.statusBar().showMessage("Запуск обучения...")
                self.training_thread = TrainingThread(
                    self.model, train_loader, val_loader, 
                    epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE
                )
                self.training_thread.progress.connect(self.update_training_progress)
                self.training_thread.finished.connect(self.training_finished)
                self.training_thread.error.connect(self.training_error)
                
                self.training_thread.start()
                
                # Обновляем UI
                self.statusBar().showMessage("Обучение модели...")
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(0)
                
                # Блокируем кнопки обучения
                self.train_button.setEnabled(False)
                self.fine_tune_button.setEnabled(False)
                
                logger.info("Поток обучения успешно запущен")
            except Exception as e:
                error_report = create_error_report(e, "Ошибка при запуске потока обучения")
                logger.error(f"Ошибка при запуске потока обучения: {str(e)}")
                QMessageBox.critical(
                    self,
                    "Ошибка при запуске обучения",
                    f"Не удалось запустить обучение: {str(e)}"
                )
                return
                
        except Exception as e:
            error_report = create_error_report(e, "Ошибка при подготовке к обучению")
            logger.error(f"Ошибка при подготовке к обучению: {str(e)}", exc_info=True)
            QMessageBox.critical(
                self,
                "Ошибка при подготовке к обучению",
                f"Не удалось подготовить данные для обучения: {str(e)}\n\n{traceback.format_exc()}"
            )
        
    def update_training_progress(self, message):
        self.statusBar().showMessage(message)
        logger.debug(f"Прогресс обучения: {message}")
        
    def training_finished(self):
        self.progress_bar.setVisible(False)
        self.train_button.setEnabled(True)
        self.statusBar().showMessage("Обучение завершено")
        logger.info("Обучение модели успешно завершено")
        QMessageBox.information(self, "Успех", "Модель успешно обучена")
        
    def training_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.train_button.setEnabled(True)
        logger.error(f"Ошибка при обучении: {error_message}")
        QMessageBox.critical(self, "Ошибка", f"Ошибка при обучении: {error_message}")
        
    def select_model(self):
        """Выбор предобученной модели"""
        try:
            model_path, _ = QFileDialog.getOpenFileName(
                self,
                "Выберите файл модели",
                MODELS_DIR,
                "Модели (*.pth)"
            )
            
            if model_path:
                logger.info(f"Выбрана модель: {model_path}")
                self.model = Wav2Vec2EmotionModel(
                    model_path=model_path,
                    num_emotions=NUM_EMOTIONS,
                    device=self.device
                )
                self.statusBar().showMessage(f"Модель загружена: {os.path.basename(model_path)}")
                QMessageBox.information(
                    self,
                    "Модель загружена",
                    f"Модель успешно загружена из файла: {os.path.basename(model_path)}"
                )
        except Exception as e:
            error_report = create_error_report(e, "Ошибка при загрузке модели")
            logger.error(f"Ошибка при загрузке модели: {str(e)}")
            QMessageBox.critical(
                self,
                "Ошибка при загрузке модели",
                f"Не удалось загрузить модель: {str(e)}\n\n{traceback.format_exc()}\n\nОтчет об ошибке сохранен: {error_report}"
            )
        
    def open_settings(self):
        """Открывает окно настроек"""
        try:
            logger.info("Открытие окна настроек")
            settings_window = SettingsWindow(self)
            if settings_window.exec() == QDialog.DialogCode.Accepted:
                # Обновляем параметры аудио процессора
                self.audio_processor.update_parameters()
                
                # Применяем новую тему
                self.apply_theme()
                
                # Обновляем размер окна
                self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
                
                # Обновляем видимость прогресс-бара
                self.progress_bar.setVisible(SHOW_PROGRESS)
                
                self.statusBar().showMessage("Настройки сохранены")
                logger.info("Настройки успешно применены")
        except Exception as e:
            error_report = create_error_report(e, "Ошибка при открытии настроек")
            logger.error(f"Ошибка при открытии настроек: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при открытии настроек: {str(e)}")
        
    def open_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите аудиофайл",
            "",
            "WAV Files (*.wav)"
        )
        if file_path:
            self.handle_audio_file(file_path)
            
    def record_audio(self):
        """Запись аудио с микрофона"""
        try:
            # Проверяем, идет ли запись
            if hasattr(self, 'recording_thread') and self.recording_thread is not None and hasattr(self.recording_thread, 'is_alive') and self.recording_thread.is_alive():
                logger.warning("Запись уже идет, останавливаем...")
                self.audio_processor.stop_recording()
                try:
                    self.recording_thread.join(timeout=2)
                except Exception as e:
                    logger.warning(f"Ошибка при остановке потока записи: {str(e)}")
                self.statusBar().showMessage("Запись остановлена")
                return
                
            logger.info("Начало записи аудио")
            
            # Изменяем внешний вид кнопки записи
            self.record_button.setText("Остановить запись")
            self.record_button.setStyleSheet("background-color: #ff0000; color: white;")
            self.statusBar().showMessage("Идет запись...")
            
            # Создаем и запускаем поток для записи
            def record_thread_func():
                try:
                    # Записываем аудио
                    self.current_audio = self.audio_processor.record_audio(duration=MAX_AUDIO_LENGTH)
                    # Сохраняем в файл
                    self.current_audio_path = os.path.join(PROCESSED_DATA_DIR, "recorded_audio.wav")
                    self.audio_processor.save_audio(self.current_audio, self.current_audio_path)
                    # Обновляем UI
                    self.update_recording_ui_completed()
                except Exception as e:
                    logger.error(f"Ошибка при записи аудио: {str(e)}")
                    # Обновляем UI в случае ошибки
                    self.update_recording_ui_error(str(e))
            
            self.recording_thread = threading.Thread(target=record_thread_func)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
        except Exception as e:
            error_report = create_error_report(e, "Ошибка при записи аудио")
            logger.error(f"Ошибка при записи аудио: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при записи аудио: {str(e)}")
            self.update_recording_ui_error(str(e))
            
    def update_recording_ui_completed(self):
        """Обновляет UI после успешной записи"""
        # Вызывается из потока записи, нужно использовать сигналы
        # или QtCore.QMetaObject.invokeMethod для обновления UI
        def update():
            self.record_button.setText("Запись аудио с микрофона")
            self.record_button.setStyleSheet("")
            self.statusBar().showMessage("Аудио записано")
            
            if self.model is not None:
                # Определяем текущую эмоцию
                try:
                    emotion_idx, probs = self.model.predict_emotion(self.current_audio)
                    emotion = list(EMOTIONS.keys())[emotion_idx]
                    self.statusBar().showMessage(f"Текущая эмоция: {emotion}")
                except Exception as e:
                    logger.error(f"Ошибка при определении эмоции: {str(e)}")
                    self.statusBar().showMessage("Аудио записано, но ошибка при определении эмоции")
                    
            logger.info("Аудио успешно записано")
            
        # Используем invokeMethod для обновления UI из другого потока
        QMetaObject.invokeMethod(self, "update_recording_ui_completed_slot", 
                                      Qt.ConnectionType.QueuedConnection)
    
    @pyqtSlot()
    def update_recording_ui_completed_slot(self):
        """Слот для обновления UI после записи"""
        self.record_button.setText("Запись аудио с микрофона")
        self.record_button.setStyleSheet("")
        self.statusBar().showMessage("Аудио записано")
        
        if self.model is not None:
            # Определяем текущую эмоцию
            try:
                emotion_idx, probs = self.model.predict_emotion(self.current_audio)
                emotion = list(EMOTIONS.keys())[emotion_idx]
                self.statusBar().showMessage(f"Текущая эмоция: {emotion}")
            except Exception as e:
                logger.error(f"Ошибка при определении эмоции: {str(e)}")
                self.statusBar().showMessage("Аудио записано, но ошибка при определении эмоции")
                
        logger.info("Аудио успешно записано")
            
    def update_recording_ui_error(self, error_msg):
        """Обновляет UI в случае ошибки при записи"""
        def update():
            self.record_button.setText("Запись аудио с микрофона")
            self.record_button.setStyleSheet("")
            self.statusBar().showMessage(f"Ошибка при записи: {error_msg}")
            
        # Используем invokeMethod для обновления UI из другого потока
        QMetaObject.invokeMethod(self, "update_recording_ui_error_slot",
                                      Qt.ConnectionType.QueuedConnection,
                                      Q_ARG(str, error_msg))
    
    @pyqtSlot(str)
    def update_recording_ui_error_slot(self, error_msg):
        """Слот для обновления UI при ошибке записи"""
        self.record_button.setText("Запись аудио с микрофона")
        self.record_button.setStyleSheet("")
        self.statusBar().showMessage(f"Ошибка при записи: {error_msg}")
        
    def process_audio(self):
        """Обработка аудио: изменение эмоции"""
        if self.model is None:
            logger.warning("Попытка обработки аудио без загруженной модели")
            QMessageBox.warning(self, "Ошибка", "Сначала выберите или обучите модель")
            return
            
        if self.current_audio is None:
            logger.warning("Попытка обработки без загруженного аудио")
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите или запишите аудио")
            return
            
        try:
            logger.info("Начало обработки аудио")
            self.statusBar().showMessage("Идет обработка аудио...")
            self.process_button.setEnabled(False)
            
            # Получаем выбранную эмоцию
            target_emotion = self.emotion_combo.currentText()
            if target_emotion not in EMOTIONS:
                logger.error(f"Неизвестная эмоция: {target_emotion}")
                QMessageBox.warning(self, "Ошибка", f"Неизвестная эмоция: {target_emotion}")
                self.process_button.setEnabled(True)
                return
                
            target_emotion_idx = EMOTIONS[target_emotion]
            logger.info(f"Целевая эмоция: {target_emotion} (индекс: {target_emotion_idx})")
            
            # Создаем и запускаем поток для обработки
            def process_thread_func():
                try:
                    # Изменяем эмоцию
                    modified_audio = self.model.change_emotion(self.current_audio, target_emotion_idx)
                    
                    # Сохраняем результат
                    if self.current_audio_path:
                        filename = os.path.basename(self.current_audio_path)
                        output_path = os.path.join(PROCESSED_DATA_DIR, f"modified_{filename}")
                    else:
                        output_path = os.path.join(PROCESSED_DATA_DIR, "modified_audio.wav")
                        
                    self.audio_processor.save_audio(modified_audio, output_path)
                    
                    # Обновляем UI
                    self.update_processing_ui_completed(output_path)
                except Exception as e:
                    logger.error(f"Ошибка при обработке аудио: {str(e)}")
                    logger.error(traceback.format_exc())
                    self.update_processing_ui_error(str(e))
            
            processing_thread = threading.Thread(target=process_thread_func)
            processing_thread.daemon = True
            processing_thread.start()
            
        except Exception as e:
            error_report = create_error_report(e, "Ошибка при обработке аудио")
            logger.error(f"Ошибка при обработке аудио: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.warning(self, "Ошибка", f"Ошибка при обработке аудио: {str(e)}")
            self.process_button.setEnabled(True)
            
    def update_processing_ui_completed(self, output_path):
        """Обновляет UI после успешной обработки"""
        QMetaObject.invokeMethod(self, "update_processing_ui_completed_slot", 
                                      Qt.ConnectionType.QueuedConnection,
                                      Q_ARG(str, output_path))
    
    @pyqtSlot(str)
    def update_processing_ui_completed_slot(self, output_path):
        """Слот для обновления UI после обработки"""
        self.process_button.setEnabled(True)
        self.statusBar().showMessage(f"Аудио обработано и сохранено")
        logger.info(f"Аудио успешно обработано и сохранено: {output_path}")
        QMessageBox.information(self, "Успех", f"Аудио успешно обработано и сохранено:\n{output_path}")
        
    def update_processing_ui_error(self, error_msg):
        """Обновляет UI в случае ошибки при обработке"""
        QMetaObject.invokeMethod(self, "update_processing_ui_error_slot",
                                      Qt.ConnectionType.QueuedConnection,
                                      Q_ARG(str, error_msg))
    
    @pyqtSlot(str)
    def update_processing_ui_error_slot(self, error_msg):
        """Слот для обновления UI при ошибке обработки"""
        self.process_button.setEnabled(True)
        self.statusBar().showMessage(f"Ошибка при обработке: {error_msg}")
        QMessageBox.warning(self, "Ошибка", f"Ошибка при обработке аудио: {error_msg}")
            
    def auto_save_model(self):
        """Автосохранение модели"""
        if AUTO_SAVE and self.model is not None:
            try:
                save_path = os.path.join(MODELS_DIR, "auto_saved_model.pth")
                self.model.save_model(save_path)
                logger.debug("Модель успешно автосохранена")
            except Exception as e:
                error_report = create_error_report(e, "Ошибка при автосохранении модели")
                logger.error(f"Ошибка автосохранения: {str(e)}")
                self.statusBar().showMessage(f"Ошибка автосохранения: {str(e)}")
                
    def closeEvent(self, event):
        """Обработка закрытия окна"""
        try:
            # Сохраняем модель перед закрытием
            if AUTO_SAVE and self.model is not None:
                save_path = os.path.join(MODELS_DIR, "auto_saved_model.pth")
                self.model.save_model(save_path)
                logger.info("Модель успешно сохранена перед закрытием")
        except Exception as e:
            error_report = create_error_report(e, "Ошибка при сохранении модели перед закрытием")
            logger.error(f"Ошибка сохранения: {str(e)}")
            self.statusBar().showMessage(f"Ошибка сохранения: {str(e)}")
        event.accept()

    def download_pretrained_model(self):
        """Скачивание предобученной модели"""
        try:
            logger.info("Начало скачивания предобученной модели")
            self.statusBar().showMessage("Скачивание модели...")
            self.download_model_button.setEnabled(False)
            
            # Создаем директорию для моделей, если её нет
            os.makedirs(MODELS_DIR, exist_ok=True)
            
            # Инициализируем модель и скачиваем веса
            self.model = Wav2Vec2EmotionModel()
            model_path = os.path.join(MODELS_DIR, "pretrained_model.pth")
            self.model.save_model(model_path)
            
            self.statusBar().showMessage("Модель успешно скачана")
            logger.info("Предобученная модель успешно скачана")
            QMessageBox.information(self, "Успех", "Предобученная модель успешно скачана")
            
        except Exception as e:
            error_report = create_error_report(e, "Ошибка при скачивании модели")
            logger.error(f"Ошибка при скачивании модели: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при скачивании модели: {str(e)}")
        finally:
            self.download_model_button.setEnabled(True)
            
    def fine_tune_model(self):
        """Дообучение предобученной модели"""
        try:
            if not os.path.exists(RAW_DATA_DIR):
                logger.error(f"Папка с датасетом не найдена: {RAW_DATA_DIR}")
                QMessageBox.warning(self, "Ошибка", "Папка с датасетом не найдена")
                return
                
            logger.info("Начало дообучения модели")
            self.statusBar().showMessage("Подготовка к дообучению...")
            self.fine_tune_button.setEnabled(False)
            
            # Загружаем предобученную модель
            model_path = os.path.join(MODELS_DIR, "pretrained_model.pth")
            if not os.path.exists(model_path):
                QMessageBox.warning(self, "Ошибка", 
                    "Предобученная модель не найдена. Сначала скачайте предобученную модель.")
                self.fine_tune_button.setEnabled(True)
                return
            
            # Создаем CSV файл с метками
            csv_path = os.path.join(DATA_DIR, "dataset.csv")
            create_dataset_csv(RAW_DATA_DIR, csv_path)
            
            # Загружаем модель
            try:
                self.model = Wav2Vec2EmotionModel(
                    model_path=model_path,
                    num_emotions=NUM_EMOTIONS,
                    device=self.device
                )
                logger.info("Предобученная модель успешно загружена")
            except Exception as e:
                error_report = create_error_report(e, "Ошибка при загрузке модели")
                logger.error(f"Ошибка при загрузке модели: {str(e)}")
                QMessageBox.critical(self, "Ошибка", 
                    f"Ошибка при загрузке модели: {str(e)}")
                self.fine_tune_button.setEnabled(True)
                return
            
            # Создаем датасет
            try:
                # Создаем экземпляр датасета с правильными параметрами
                dataset = EmotionDataset(
                    csv_path=csv_path,
                    data_dir=RAW_DATA_DIR,
                    sample_rate=SAMPLE_RATE
                )
                
                # Если датасет пустой, выводим ошибку
                if len(dataset) == 0:
                    QMessageBox.warning(self, "Пустой датасет", 
                        "Датасет не содержит примеров для обучения. Добавьте аудиофайлы в папку data/raw.")
                    self.fine_tune_button.setEnabled(True)
                    return
                
                # Разделяем на обучающую и валидационную выборки
                val_size = int(len(dataset) * VALIDATION_SPLIT)
                train_size = len(dataset) - val_size
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
                
                # Создаем загрузчики данных
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
                
                # Запускаем дообучение в отдельном потоке
                self.training_thread = TrainingThread(
                    self.model,
                    train_loader,
                    val_loader,
                    NUM_EPOCHS,
                    LEARNING_RATE * 0.1  # Используем меньший learning rate для дообучения
                )
                
                self.training_thread.progress.connect(self.update_training_progress)
                self.training_thread.finished.connect(self.training_finished)
                self.training_thread.error.connect(self.training_error)
                
                self.progress_bar.setVisible(SHOW_PROGRESS)
                self.progress_bar.setRange(0, NUM_EPOCHS)
                self.progress_bar.setValue(0)
                
                # Блокируем кнопки обучения
                self.train_button.setEnabled(False)
                
                self.training_thread.start()
                self.statusBar().showMessage("Дообучение модели...")
                
            except Exception as e:
                error_report = create_error_report(e, "Ошибка при дообучении модели")
                logger.error(f"Ошибка при дообучении модели: {str(e)}", exc_info=True)
                QMessageBox.critical(self, "Ошибка", 
                    f"Ошибка при дообучении модели: {str(e)}")
                self.fine_tune_button.setEnabled(True)
                
        except Exception as e:
            error_report = create_error_report(e, "Общая ошибка при дообучении")
            logger.error(f"Общая ошибка при дообучении: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Ошибка", 
                f"Ошибка при дообучении модели: {str(e)}")
            self.fine_tune_button.setEnabled(True) 
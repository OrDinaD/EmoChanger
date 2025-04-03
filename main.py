import sys
import os
import traceback
from PyQt6.QtWidgets import QApplication, QMessageBox
from src.ui.main_window import MainWindow
import torch
import logging

# Настройка расширенного логирования
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/emochanger_debug.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('EmoChanger')

# Глобальная переменная для хранения устройства
device = None

def excepthook(exc_type, exc_value, exc_traceback):
    """Глобальный обработчик исключений"""
    error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logger.critical(f"Необработанное исключение: {error_msg}")
    
    # Показываем диалог с ошибкой
    if QApplication.instance():
        QMessageBox.critical(None, "Критическая ошибка", 
                            f"Произошла критическая ошибка:\n\n{str(exc_value)}\n\n"
                            f"Подробная информация записана в logs/emochanger_debug.log")
    
    # Записываем ошибку в файл для последующего анализа
    with open('logs/crash_report.txt', 'w', encoding='utf-8') as f:
        f.write(error_msg)

def main():
    try:
        global device
        logger.info("============= Запуск EmoChanger =============")
        logger.info(f"Текущая директория: {os.getcwd()}")
        logger.info(f"Путь к исполняемому файлу: {sys.executable}")
        logger.info(f"Аргументы запуска: {sys.argv}")
        logger.info(f"Версия Python: {sys.version}")
        
        # Получаем версию PyQt6 безопасным способом
        try:
            from PyQt6.QtCore import QT_VERSION_STR
            pyqt_version = QT_VERSION_STR
        except:
            pyqt_version = "неизвестно"
        logger.info(f"Версия PyQt: {pyqt_version}")
        
        # Проверяем структуру директорий
        logger.info("Проверка структуры директорий:")
        for dir_path in ['data', 'src', 'configs']:
            exists = os.path.exists(dir_path)
            logger.info(f"  - {dir_path}: {'существует' if exists else 'не существует'}")
        
        # Логируем информацию о доступных устройствах
        logger.info("Информация о доступных устройствах:")
        logger.info(f"CUDA доступна: {torch.cuda.is_available()}")
        if hasattr(torch.backends, 'mps'):
            logger.info(f"MPS доступен: {torch.backends.mps.is_available()}")
        
        # Проверяем наличие Metal вместо CUDA на Mac M1
        if sys.platform == "darwin" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info(f"Using Apple Silicon MPS device: {device}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}")
        
        # Создаем приложение
        app = QApplication(sys.argv)
        logger.info("QApplication инициализирован")
        
        # Создаем и показываем главное окно
        logger.info("Создание главного окна...")
        window = MainWindow(device=device)
        logger.info("Показ главного окна...")
        window.show()
        
        # Запускаем главный цикл приложения
        logger.info("Запуск главного цикла приложения...")
        exit_code = app.exec()
        logger.info(f"Приложение завершило работу с кодом: {exit_code}")
        return exit_code
        
    except Exception as e:
        logger.critical(f"Ошибка при запуске: {str(e)}", exc_info=True)
        # Записываем ошибку в файл
        with open('logs/startup_error.txt', 'w', encoding='utf-8') as f:
            f.write(f"Ошибка при запуске: {str(e)}\n\n")
            f.write(traceback.format_exc())
        
        # Если уже создано приложение, показываем сообщение об ошибке
        if QApplication.instance():
            QMessageBox.critical(None, "Ошибка при запуске", 
                               f"Не удалось запустить приложение:\n\n{str(e)}\n\n"
                               "Проверьте файл logs/startup_error.txt для получения подробной информации.")
        return 1

if __name__ == "__main__":
    # Устанавливаем глобальный обработчик исключений
    sys.excepthook = excepthook
    sys.exit(main()) 
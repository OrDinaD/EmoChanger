import os
import logging
import traceback
import sys
from datetime import datetime
from ..configs.config import LOGS_DIR, LOG_FORMAT

def setup_logger():
    # Создаем директорию для логов, если её нет
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    logger = logging.getLogger('EmoChanger')
    logger.setLevel(logging.DEBUG)
    
    # Убедимся, что нет существующих обработчиков (для предотвращения дублирования)
    if logger.handlers:
        logger.handlers = []
    
    # Создаем форматтер для логов
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Создаем файл для логов с текущей датой
    log_file = os.path.join(
        LOGS_DIR,
        f"emochanger_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    # Добавляем обработчик для файла
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Добавляем обработчик для консоли
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Выводим основную информацию о системе
    logger.info("="*50)
    logger.info(f"Запуск EmoChanger в {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Операционная система: {sys.platform}")
    logger.info(f"Python версия: {sys.version}")
    logger.info(f"Путь к логу: {log_file}")
    logger.info("="*50)
    
    # Устанавливаем обработчик для непойманных исключений
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Стандартная обработка для KeyboardInterrupt
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
            
        logger.critical("Непойманное исключение:", exc_info=(exc_type, exc_value, exc_traceback))
        
    sys.excepthook = handle_exception
    
    return logger

# Функция для создания отчета об ошибке
def create_error_report(error: Exception, context: str = ""):
    """Создает подробный отчет об ошибке"""
    # Создаем директорию для логов, если её нет
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context,
        'traceback': traceback.format_exc(),
        'platform': sys.platform,
        'python_version': sys.version
    }
    
    # Сохраняем отчет в файл
    report_file = os.path.join(
        LOGS_DIR,
        f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(report_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(report, f, ensure_ascii=False, indent=4)
    
    # Выводим сообщение для пользователя
    print(f"Произошла ошибка: {str(error)}")
    print(f"Отчет сохранен в {report_file}")
    
    return report_file 
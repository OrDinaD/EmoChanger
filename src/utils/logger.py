import os
import logging
import traceback
from datetime import datetime
from ..configs.config import LOGS_DIR, LOG_FORMAT

def setup_logger():
    logger = logging.getLogger('EmoChanger')
    logger.setLevel(logging.DEBUG)
    
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
    
    return logger

# Функция для создания отчета об ошибке
def create_error_report(error: Exception, context: str = ""):
    """Создает подробный отчет об ошибке"""
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context,
        'traceback': traceback.format_exc()
    }
    
    # Сохраняем отчет в файл
    report_file = os.path.join(
        LOGS_DIR,
        f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(report_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(report, f, ensure_ascii=False, indent=4)
    
    return report_file 
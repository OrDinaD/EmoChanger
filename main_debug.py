import sys
import os
import traceback
from PyQt6.QtWidgets import QApplication
import logging

# Настраиваем логирование
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main_debug")

def main():
    try:
        # Проверяем наличие CUDA
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Импортируем с проверкой
        try:
            from src.ui.main_window import MainWindow
            logger.info("MainWindow успешно импортирован")
        except Exception as e:
            logger.error(f"Ошибка импорта MainWindow: {str(e)}")
            logger.error(traceback.format_exc())
            return
        
        # Создаем приложение
        app = QApplication(sys.argv)
        
        # Создаем и показываем главное окно
        try:
            logger.info("Создание MainWindow...")
            window = MainWindow()
            logger.info("MainWindow создан успешно")
            window.show()
            logger.info("MainWindow показан")
        except Exception as e:
            logger.error(f"Ошибка создания или отображения окна: {str(e)}")
            logger.error(traceback.format_exc())
            return
        
        # Запускаем главный цикл приложения
        logger.info("Запуск главного цикла приложения")
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"Непредвиденная ошибка: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    logger.info("Запуск приложения")
    main() 
#!/bin/bash
echo "Настройка EmoChanger для Mac с Apple Silicon"

# Создаем виртуальное окружение
python3 -m venv venv
source venv/bin/activate

# Устанавливаем зависимости
pip install --upgrade pip
pip install -r requirements.txt

# Проверка поддержки MPS
python3 -c "import torch; print(f'MPS доступен: {torch.backends.mps.is_available()}')"

echo "Настройка завершена! Для запуска приложения используйте команду:"
echo "source venv/bin/activate && python main.py"

echo "Для сборки приложения используйте команду:"
echo "source venv/bin/activate && pyinstaller EmoChanger.spec"

echo "Для создания DMG-образа используйте:"
echo "./create_dmg.sh" 
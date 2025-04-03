#!/bin/bash
echo "Создание DMG-образа для EmoChanger"

# Проверяем, что приложение собрано
if [ ! -d "dist/EmoChanger.app" ]; then
    echo "Ошибка: приложение не найдено в директории dist/"
    echo "Сначала соберите приложение командой: pyinstaller EmoChanger.spec"
    exit 1
fi

# Проверяем, есть ли hdiutil (доступен только на macOS)
if ! command -v hdiutil &> /dev/null; then
    echo "Ошибка: команда hdiutil не найдена. Убедитесь, что вы используете macOS."
    exit 1
fi

# Создаем временную директорию для DMG
TMP_DIR=$(mktemp -d)
DMG_DIR="$TMP_DIR/EmoChanger"
mkdir -p "$DMG_DIR"

# Копируем приложение
cp -R "dist/EmoChanger.app" "$DMG_DIR/"

# Создаем ссылку на папку Applications
ln -s /Applications "$DMG_DIR/Applications"

# Создаем DMG-образ
DMG_PATH="dist/EmoChanger_Installer.dmg"
hdiutil create -volname "EmoChanger Installer" -srcfolder "$DMG_DIR" -ov -format UDZO "$DMG_PATH"

# Очищаем временную директорию
rm -rf "$TMP_DIR"

echo "DMG-образ создан: $DMG_PATH"
echo "Теперь вы можете установить приложение, открыв этот файл и перетащив EmoChanger.app в папку Applications." 
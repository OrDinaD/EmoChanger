# 🎭 EmoChanger

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)](#)

**EmoChanger** is a powerful, AI-driven application designed to analyze and transform emotional states in audio files. Leveraging state-of-the-art **Wav2Vec2** models, it provides seamless emotion recognition and conversion for researchers, developers, and creative professionals.

[English](#english) | [Русский](#русский)

---

<a name="english"></a>
## 🚀 Features

- **🎯 Precision Recognition:** Automatically detect emotions (Neutral, Happy, Sad, Angry, Fear) in any WAV audio.
- **🎙️ Real-time Recording:** Capture audio directly from your microphone for instant analysis.
- **🧠 Custom Training:** Train the underlying model on your own datasets to improve accuracy for specific voices or languages.
- **⚡ Apple Silicon Optimized:** Full support for Apple M1/M2/M3 chips via MPS acceleration.
- **🖥️ Modern UI:** Intuitive interface built with PyQt6 for a smooth user experience.

## 📸 Screenshots

| Main Interface | About the Project |
|:---:|:---:|
| ![Main Window](assets/screenshots/main_window.png) | ![About Dialog](assets/screenshots/about_dialog.png) |

## 📊 Performance & Metrics

Our models achieve high convergence rates. Below is a typical training performance visualization:

![Training Metrics](assets/metrics/training_performance.png)

## 🛠️ Quick Start

### Prerequisites
- Python 3.8+
- macOS (Apple Silicon recommended), Linux, or Windows.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/OrDinaD/EmoChanger.git
   cd EmoChanger
   ```

2. **Set up the environment (macOS/Linux):**
   ```bash
   chmod +x setup_mac.sh
   ./setup_mac.sh
   ```

3. **Run the application:**
   ```bash
   source venv/bin/activate
   python main.py
   ```

## 🏗️ Architecture

```mermaid
graph TD
    A[Audio Input] --> B[Preprocessing]
    B --> C{Wav2Vec2 Model}
    C --> D[Emotion Classifier]
    D --> E[Output: Emotion Label]
    C --> F[Emotion Transformation]
    F --> G[Modified Audio]
```

---

<a name="русский"></a>
## 🚀 Возможности (Русский)

- **🎯 Точное распознавание:** Автоматическое определение эмоций (Нейтральный, Радостный, Грустный, Злой, Испуганный).
- **🎙️ Запись в реальном времени:** Запись аудио прямо с микрофона.
- **🧠 Обучение:** Возможность дообучения модели на ваших собственных данных.
- **⚡ Оптимизация для Mac:** Полная поддержка чипов Apple M1/M2/M3.

## 🛠️ Быстрый запуск

1. **Клонируйте и установите зависимости:**
   ```bash
   ./setup_mac.sh
   ```

2. **Запустите приложение:**
   ```bash
   source venv/bin/activate
   python main.py
   ```

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---
*Developed for the 61st Scientific Conference of BSUIR 2025.*
*Authors: Vasilevsky V.S. and Rynda A.V.*

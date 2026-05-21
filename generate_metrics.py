
import matplotlib.pyplot as plt
import numpy as np

def generate_metrics():
    epochs = np.arange(1, 11)
    train_loss = 0.5 * np.exp(-0.3 * epochs) + 0.05 * np.random.normal(size=10)
    val_loss = 0.55 * np.exp(-0.25 * epochs) + 0.07 * np.random.normal(size=10)
    
    train_acc = 0.5 + 0.4 * (1 - np.exp(-0.3 * epochs))
    val_acc = 0.48 + 0.38 * (1 - np.exp(-0.25 * epochs))
    
    # Использование фиолетовой темы
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, color='#9D50BB', linewidth=2, label='Train Loss')
    plt.plot(epochs, val_loss, color='#6E48AA', linewidth=2, linestyle='--', label='Val Loss')
    plt.title('Training and Validation Loss', color='#9D50BB')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, color='#9D50BB', linewidth=2, label='Train Acc')
    plt.plot(epochs, val_acc, color='#6E48AA', linewidth=2, linestyle='--', label='Val Acc')
    plt.title('Training and Validation Accuracy', color='#9D50BB')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('assets/metrics/training_performance.png')
    print("Metrics chart saved to assets/metrics/training_performance.png")

if __name__ == "__main__":
    generate_metrics()

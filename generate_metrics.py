
import matplotlib.pyplot as plt
import numpy as np

def generate_metrics():
    epochs = np.arange(1, 11)
    train_loss = 0.5 * np.exp(-0.3 * epochs) + 0.05 * np.random.normal(size=10)
    val_loss = 0.55 * np.exp(-0.25 * epochs) + 0.07 * np.random.normal(size=10)
    
    train_acc = 0.5 + 0.4 * (1 - np.exp(-0.3 * epochs))
    val_acc = 0.48 + 0.38 * (1 - np.exp(-0.25 * epochs))
    
    # Использование светлой темы в стиле Apple
    plt.style.use('default')
    plt.figure(figsize=(10, 5), facecolor='#F5F5F7')
    
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_facecolor('#FFFFFF')
    plt.plot(epochs, train_loss, color='#5E5CE6', linewidth=2.5, label='Train Loss')
    plt.plot(epochs, val_loss, color='#AF52DE', linewidth=2.5, linestyle=':', label='Val Loss')
    plt.title('Training and Validation Loss', color='#1D1D1F', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs', color='#86868B')
    plt.ylabel('Loss', color='#86868B')
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.15, color='#86868B')
    
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_facecolor('#FFFFFF')
    plt.plot(epochs, train_acc, color='#5E5CE6', linewidth=2.5, label='Train Acc')
    plt.plot(epochs, val_acc, color='#AF52DE', linewidth=2.5, linestyle=':', label='Val Acc')
    plt.title('Training and Validation Accuracy', color='#1D1D1F', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs', color='#86868B')
    plt.ylabel('Accuracy', color='#86868B')
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.15, color='#86868B')
    
    plt.tight_layout()
    plt.savefig('assets/metrics/training_performance.png', dpi=300)
    print("Metrics chart saved to assets/metrics/training_performance.png")

if __name__ == "__main__":
    generate_metrics()

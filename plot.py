import matplotlib.pyplot as plt

from constants import BATCH_SIZES, EPOCHS


def plot_accuracies(accuracies: dict) -> None:
    plt.figure(figsize=(10, 6))

    for epochs, accuracy in accuracies.items():
        plt.plot(accuracy, label=f'Epochs: {epochs}')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Epoch for Different Epochs Number')
    plt.legend()
    plt.show()


def plot_accuracies_and_training_times(accuracies, training_times):
    fig, ax1 = plt.subplots()

    for batch_size in BATCH_SIZES:
        accuracies_list = [accuracies[(batch_size, epochs)] for epochs in EPOCHS]
        ax1.plot(EPOCHS, accuracies_list, label=f'Batch size {batch_size}')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')

    ax2 = ax1.twinx()
    for batch_size in BATCH_SIZES:
        training_times_list = [training_times[(batch_size, epochs)] for epochs in EPOCHS]
        ax2.plot(EPOCHS, training_times_list, linestyle='dashed', label=f'Training time (Batch size {batch_size})')
        ax2.set_ylabel('Training Time (seconds)')

    fig.legend(loc='upper left')
    plt.title('Training Accuracy and Time for Different Batch Sizes and Epochs')
    plt.show()

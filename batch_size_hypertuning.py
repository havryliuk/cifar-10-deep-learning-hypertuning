from data import prepare_and_get_data
from model import get_model
from plot import plot_accuracies

BATCH_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048]

(train_images, train_labels), (test_images, test_labels) = prepare_and_get_data()

accuracies = {}
for batch_size in BATCH_SIZES:
    model = get_model()
    print(f'Training with batch size: {batch_size}')
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels),
                        batch_size=batch_size)
    accuracies[batch_size] = history.history['accuracy']

plot_accuracies(accuracies)

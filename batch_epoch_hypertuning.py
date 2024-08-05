import itertools
import time

from constants import BATCH_SIZES, EPOCHS
from data import prepare_and_get_data
from model import get_model
from plot import plot_accuracies_and_training_times


(train_images, train_labels), (test_images, test_labels) = prepare_and_get_data()

accuracies = {}
training_times = {}

for batch_size, epochs in itertools.product(BATCH_SIZES, EPOCHS):
    model = get_model()

    print(f'Training with {batch_size} batches, {epochs} epochs')
    start_time = time.time()
    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels),
                        batch_size=batch_size)
    training_time = time.time() - start_time
    accuracy = history.history['accuracy'][-1]

    accuracies[(batch_size, epochs)] = accuracy
    training_times[(batch_size, epochs)] = training_time

most_effective = max(accuracies, key=lambda k: (accuracies[k], -training_times[k]))
print(f'Most effective training: Batch size {most_effective[0]}, Epochs {most_effective[1]}, '
      f'Accuracy {accuracies[most_effective]}, Training time {training_times[most_effective]} seconds')

plot_accuracies_and_training_times(accuracies, training_times)

from data import prepare_and_get_data
from model import get_model
from plot import plot_accuracies

(train_images, train_labels), (test_images, test_labels) = prepare_and_get_data()

accuracies = {}
for epochs in [30, 40, 50]:
    model = get_model()
    print(f'Training with epochs: {epochs}')
    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))
    accuracies[epochs] = history.history['accuracy']

plot_accuracies(accuracies)

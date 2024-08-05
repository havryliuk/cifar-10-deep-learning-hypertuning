import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data import prepare_and_get_data

model = tf.keras.models.load_model('32_50_model.h5')

(train_images, train_labels), (test_images, test_labels) = prepare_and_get_data()

random_index = random.randint(0, len(test_images) - 1)
print(f'Selected random image at index {random_index}')
random_image = test_images[random_index]
plt.imshow(random_image)
plt.show()

random_label = test_labels[random_index]

prediction = model.predict(np.expand_dims(random_image, axis=0))
print(f'Prediction: {prediction}')
predicted_label = np.argmax(prediction)

is_correct = predicted_label == random_label
print(f"Predicted label: {predicted_label}, True label: {random_label}, Correct: {is_correct}")

from data import prepare_and_get_data
from model import get_model

(train_images, train_labels), (test_images, test_labels) = prepare_and_get_data()

model = get_model()
history = model.fit(train_images, train_labels, epochs=50, validation_data=(test_images, test_labels), batch_size=32)

model.save('32_50_model.keras')

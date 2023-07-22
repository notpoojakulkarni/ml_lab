
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Step 2: Build the neural network model
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # Flatten the 28x28 input image to a 1D array
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Step 3: Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
epochs = 10
batch_size = 128
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Step 5: Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Step 6: Plot the training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Step 7: Make predictions
num_samples_to_predict = 5
sample_indices = np.random.randint(0, x_test.shape[0], num_samples_to_predict)
for index in sample_indices:
    sample_image = x_test[index]
    true_label = np.argmax(y_test[index])
    prediction = np.argmax(model.predict(np.expand_dims(sample_image, axis=0)))
    plt.imshow(sample_image, cmap='gray')
    plt.title(f'True Label: {true_label}, Prediction: {prediction}')
    plt.show()

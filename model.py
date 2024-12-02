import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input

class DigitRecognitionModel:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        """Build and return the CNN model for digit recognition"""
        model = Sequential([
            Input(shape=(28, 28, 1)),  # Explicit input layer
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X_train, y_train, X_test, y_test, epochs=5, batch_size=32):
        """Train the digit recognition model"""
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'best_digit_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[checkpoint]
        )
        return history

    def predict(self, image):
        """Make predictions on a single digit image"""
        return self.model.predict(image, verbose=0)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from Evaluation import evaluation


def Model(X, Y, test_x, test_y):
    model = Sequential([
        Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(X.shape[1], X.shape2, X.shape[3])),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(Y.shape[0], activation='relu'),
        Dropout(0.5),
        Dense(Y.shape[1], activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Summary of the model
    model.summary()

    # Train the model
    model.fit(X, Y, epochs=100, steps_per_epoch=50, batch_size=4)
    pred = model.predict(test_x)

    return pred


def Model_DCNN(train_data, train_target, test_data, test_target):
    pred = Model(train_data, train_target, test_data, test_target)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(test_target, pred)

    return Eval, pred


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from Evaluation import net_evaluation


# -------------------------------------------------
# Build FCN Model
# -------------------------------------------------
def build_fcn(input_shape=(256, 256, 3), num_classes=1):

    inputs = Input(shape=input_shape)

    # ---------------- Encoder (VGG-like) ----------------
    # Block 1
    x = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    # Block 2
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    # Block 3
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    # Block 4
    x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    # Block 5
    x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    # ---------------- Fully Convolutional Layers ----------------
    x = Conv2D(4096, (7,7), activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(4096, (1,1), activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)

    # Classifier
    x = Conv2D(num_classes, (1,1), padding='same')(x)

    # ---------------- Upsampling (FCN-32s) ----------------
    outputs = Conv2DTranspose(
        num_classes,
        kernel_size=(64,64),
        strides=(32,32),
        padding='same',
        activation='sigmoid' if num_classes == 1 else 'softmax'
    )(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(1e-4),
        loss='binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# -------------------------------------------------
# Model Wrapper
# -------------------------------------------------
def Model_FCN(Image, GT):

    fcn_model = build_fcn()
    fcn_model.summary()

    Pred = fcn_model.predict(Image)
    eval = net_evaluation(Pred, GT)

    return eval, Pred

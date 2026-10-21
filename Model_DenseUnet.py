import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Concatenate, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from Evaluation import net_evaluation


# -------------------------------------------------
# Dense Block
# -------------------------------------------------
def dense_block(x, num_layers, growth_rate):
    for _ in range(num_layers):
        y = BatchNormalization()(x)
        y = Activation('relu')(y)
        y = Conv2D(growth_rate, (3, 3), padding='same')(y)
        x = Concatenate()([x, y])
    return x


def transition_down(x, filters):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    return x


def transition_up(x, filters):
    x = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(x)
    return x


def build_denseunet(input_shape=(256, 256, 3), num_classes=1,
                    growth_rate=32):

    inputs = Input(shape=input_shape)

    # -------- Encoder --------
    d1 = dense_block(inputs, 4, growth_rate)
    p1 = transition_down(d1, 128)

    d2 = dense_block(p1, 4, growth_rate)
    p2 = transition_down(d2, 256)

    d3 = dense_block(p2, 4, growth_rate)
    p3 = transition_down(d3, 512)

    d4 = dense_block(p3, 4, growth_rate)
    p4 = transition_down(d4, 1024)

    # -------- Bottleneck --------
    bn = dense_block(p4, 4, growth_rate)

    # -------- Decoder --------
    u4 = transition_up(bn, 512)
    u4 = Concatenate()([u4, d4])
    d5 = dense_block(u4, 4, growth_rate)

    u3 = transition_up(d5, 256)
    u3 = Concatenate()([u3, d3])
    d6 = dense_block(u3, 4, growth_rate)

    u2 = transition_up(d6, 128)
    u2 = Concatenate()([u2, d2])
    d7 = dense_block(u2, 4, growth_rate)

    u1 = transition_up(d7, 64)
    u1 = Concatenate()([u1, d1])
    d8 = dense_block(u1, 4, growth_rate)

    # -------- Output --------
    outputs = Conv2D(
        num_classes,
        (1, 1),
        activation='sigmoid' if num_classes == 1 else 'softmax'
    )(d8)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(1e-4),
        loss='binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def Model_DenseUNet(Image, GT):

    model = build_denseunet()
    model.summary()

    Pred = model.predict(Image)
    eval = net_evaluation(Pred, GT)

    return eval, Pred



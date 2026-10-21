import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, GlobalAveragePooling2D,
                                     Dense, Reshape, Multiply, Add,
                                     Activation, BatchNormalization,
                                     GlobalMaxPooling2D, Concatenate,
                                     LayerNormalization, MultiHeadAttention,
                                     Dropout, Flatten)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from Evaluation import evaluation


def adaptive_se_block(x, reduction=16):
    filters = x.shape[-1]

    se = GlobalAveragePooling2D()(x)
    se = Dense(filters // reduction, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, 1, filters))(se)

    x = Multiply()([x, se])
    return x


def channel_attention(x, reduction=16):
    filters = x.shape[-1]

    avg_pool = GlobalAveragePooling2D()(x)
    max_pool = GlobalMaxPooling2D()(x)

    dense_1 = Dense(filters // reduction, activation='relu')
    dense_2 = Dense(filters, activation='sigmoid')

    avg_out = dense_2(dense_1(avg_pool))
    max_out = dense_2(dense_1(max_pool))

    scale = Add()([avg_out, max_out])
    scale = Reshape((1, 1, filters))(scale)

    return Multiply()([x, scale])


def spatial_attention(x):
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(x, axis=-1, keepdims=True)

    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    attention = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)

    return Multiply()([x, attention])


def vit_block(x, num_heads=4, projection_dim=64):
    shape = x.shape
    h, w, c = shape[1], shape[2], shape[3]

    x_flat = Reshape((h * w, c))(x)

    attn = MultiHeadAttention(num_heads=num_heads,
                              key_dim=projection_dim)(x_flat, x_flat)
    x_flat = Add()([x_flat, attn])
    x_flat = LayerNormalization()(x_flat)

    mlp = Dense(projection_dim, activation='gelu')(x_flat)
    mlp = Dense(c)(mlp)

    x_flat = Add()([x_flat, mlp])
    x_flat = LayerNormalization()(x_flat)

    x = Reshape((h, w, c))(x_flat)
    return x


def build_csavit_asenet(sol, input_shape=(20, 20, 1), num_classes=2):
    act = ['relu', 'linear', 'sigmoid', 'tanh', 'softmax']

    inputs = Input(shape=input_shape)

    # Stem
    x = Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)

    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Adaptive SE
    x = adaptive_se_block(x)

    # Channel + Spatial Attention
    x = channel_attention(x)
    x = spatial_attention(x)

    # ViT Block
    x = vit_block(x)

    # Classification Head
    x = GlobalAveragePooling2D()(x)
    x = Dense(int(sol[0]), activation=act[int(sol[2])])(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(sol[1]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def Model_CSAViT_SeNet(train_data, train_target, test_data, test_target, sol=None):
    if sol is None:
        sol = [5, 0.01, 1]
    IMG_SIZE = 20

    Train_X = train_data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    Test_X = test_data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    model = build_csavit_asenet(sol, input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=train_target.shape[1])

    model.summary()

    model.fit(Train_X, train_target,
              epochs=10,
              batch_size=32,
              validation_data=(Test_X, test_target),
              verbose=1)

    pred = model.predict(Test_X)

    Eval = evaluation(pred, test_target)

    return Eval, pred

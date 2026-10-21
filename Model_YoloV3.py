import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Add, ZeroPadding2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from Evaluation import net_evaluation


# -------------------------------------------------
# Basic Conv Block
# -------------------------------------------------
def conv_block(x, filters, kernel_size, strides=1):
    if strides == 2:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)

    x = Conv2D(filters, kernel_size, strides=strides,
               padding='same' if strides == 1 else 'valid',
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


# -------------------------------------------------
# Residual Block
# -------------------------------------------------
def residual_block(x, filters):
    shortcut = x
    x = conv_block(x, filters // 2, 1)
    x = conv_block(x, filters, 3)
    x = Add()([shortcut, x])
    return x


# -------------------------------------------------
# Darknet-53 Backbone
# -------------------------------------------------
def darknet53(inputs):
    x = conv_block(inputs, 32, 3)
    x = conv_block(x, 64, 3, strides=2)

    x = residual_block(x, 64)

    x = conv_block(x, 128, 3, strides=2)
    for _ in range(2):
        x = residual_block(x, 128)

    x = conv_block(x, 256, 3, strides=2)
    for _ in range(8):
        x = residual_block(x, 256)
    route_1 = x  # 52x52

    x = conv_block(x, 512, 3, strides=2)
    for _ in range(8):
        x = residual_block(x, 512)
    route_2 = x  # 26x26

    x = conv_block(x, 1024, 3, strides=2)
    for _ in range(4):
        x = residual_block(x, 1024)

    return route_1, route_2, x


# -------------------------------------------------
# YOLO Detection Head
# -------------------------------------------------
def yolo_head(x, filters, num_classes):
    x = conv_block(x, filters, 1)
    x = conv_block(x, filters * 2, 3)
    x = conv_block(x, filters, 1)
    x = conv_block(x, filters * 2, 3)
    x = conv_block(x, filters, 1)

    y = conv_block(x, filters * 2, 3)
    y = Conv2D(3 * (num_classes + 5), 1, padding='same')(y)

    return x, y


# -------------------------------------------------
# Build YOLOv3 Model
# -------------------------------------------------
def build_yolov3(input_shape=(416, 416, 3), num_classes=80):
    inputs = Input(shape=input_shape)

    route_1, route_2, x = darknet53(inputs)

    # First scale (13x13)
    x, y1 = yolo_head(x, 512, num_classes)

    # Second scale (26x26)
    x = conv_block(x, 256, 1)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, route_2])
    x, y2 = yolo_head(x, 256, num_classes)

    # Third scale (52x52)
    x = conv_block(x, 128, 1)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, route_1])
    _, y3 = yolo_head(x, 128, num_classes)

    model = Model(inputs, [y1, y2, y3])
    model.compile(
        optimizer=Adam(1e-4),
        loss='binary_crossentropy'  # Replace with YOLO loss for real training
    )

    return model


# -------------------------------------------------
# Model Wrapper
# -------------------------------------------------
def Model_YoloV3(Image, GT, num_classes=80):
    model = build_yolov3(num_classes=num_classes)
    model.summary()

    Pred = model.predict(Image)
    eval = net_evaluation(Pred, GT)

    return eval, Pred

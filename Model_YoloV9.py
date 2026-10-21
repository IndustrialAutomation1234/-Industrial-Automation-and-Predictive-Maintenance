import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization
from tensorflow.keras.layers import LeakyReLU, Add, Concatenate
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from Evaluation import net_evaluation


def conv_block(x, filters, kernel=3, stride=1):
    x = Conv2D(filters, kernel, strides=stride, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def gelan_block(x, filters):
    shortcut = x

    x1 = conv_block(x, filters, 1)
    x2 = conv_block(x1, filters, 3)
    x3 = conv_block(x2, filters, 3)

    x = Concatenate()([x1, x2, x3])

    x = conv_block(x, filters, 1)

    x = Add()([shortcut, x])
    return x


def yolov9_backbone(inputs):

    x = conv_block(inputs, 32, 3, 2)
    x = conv_block(x, 64, 3, 2)
    x = gelan_block(x, 64)

    x = conv_block(x, 128, 3, 2)
    x = gelan_block(x, 128)

    route_small = x

    x = conv_block(x, 256, 3, 2)
    x = gelan_block(x, 256)

    route_medium = x

    x = conv_block(x, 512, 3, 2)
    x = gelan_block(x, 512)

    route_large = x

    return route_small, route_medium, route_large


def detection_head(x, num_classes):

    x = conv_block(x, 256, 3)
    output = Conv2D(
        3 * (num_classes + 5),
        1,
        padding='same'
    )(x)

    return output


def build_yolov9(input_shape=(640, 640, 3), num_classes=80):

    inputs = Input(shape=input_shape)

    route_small, route_medium, route_large = yolov9_backbone(inputs)

    # Large object detection
    y1 = detection_head(route_large, num_classes)

    # Medium scale
    x = UpSampling2D(2)(route_large)
    x = Concatenate()([x, route_medium])
    y2 = detection_head(x, num_classes)

    # Small scale
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, route_small])
    y3 = detection_head(x, num_classes)

    model = Model(inputs, [y1, y2, y3])

    model.compile(
        optimizer=Adam(1e-4),
        loss='binary_crossentropy'  # Replace with YOLO loss for real detection
    )

    return model


def Model_YOLOv9(Image, GT, num_classes=80):

    model = build_yolov9(num_classes=num_classes)
    model.summary()

    Pred = model.predict(Image)
    eval = net_evaluation(Pred, GT)

    return eval, Pred

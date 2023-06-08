import tensorflow as tf


def layer_norm(x):
    return tf.keras.layers.LayerNormalization()(x)


def lrelu(x, alpha=0.2):
    return tf.keras.layers.LeakyReLU(alpha=alpha)(x)


def Conv2D(inputs, filters, kernel_size=3, strides=1, padding='same', use_bias=False, activation=None):
    return tf.keras.layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  use_bias=use_bias,
                                  padding=padding,
                                  activation=activation)(inputs)


def Conv2DNormLReLU(inputs, filters, kernel_size=3, strides=1, padding='same', use_bias=False):
    x = Conv2D(inputs, filters, kernel_size, strides, padding=padding, use_bias=use_bias)
    x = layer_norm(x)
    return lrelu(x)


def dwise_conv(input, kernel_size=3, channel_multiplier=1, strides=1, padding='same', use_bias=True):
    return tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding,
                                           depth_multiplier=channel_multiplier,
                                           use_bias=use_bias)(input)


def InvertedRes_block(input, expansion_ratio, output_dim, stride, use_bias=False):
    # pw
    bottleneck_dim = round(expansion_ratio * input.get_shape().as_list()[-1])
    net = Conv2DNormLReLU(input, bottleneck_dim, kernel_size=1, use_bias=use_bias)

    # dw
    net = dwise_conv(net)
    net = layer_norm(net)
    net = lrelu(net)

    # pw & linear
    net = Conv2D(net, output_dim, kernel_size=1)
    net = layer_norm(net)

    # element wise add, only for stride==1
    if (int(input.get_shape().as_list()[-1]) == output_dim) and stride == 1:
        net = input + net

    return net


def Unsample(inputs, filters, kernel_size=3):
    new_size = (2 * tf.shape(inputs)[1], 2 * tf.shape(inputs)[2])
    inputs = tf.image.resize(inputs, new_size)

    return Conv2DNormLReLU(filters=filters, kernel_size=kernel_size, inputs=inputs)


def generator(image_size=(64, 64), channels=3, kernel_size=3, init_filters=32):
    input1 = tf.keras.layers.Input((image_size[0], image_size[1], channels))

    x = input1

    x = Conv2DNormLReLU(x, init_filters, kernel_size=7)
    x = Conv2DNormLReLU(x, init_filters * 2, kernel_size=kernel_size, strides=2)
    x = Conv2DNormLReLU(x, init_filters * 2, kernel_size=kernel_size)

    x = Conv2DNormLReLU(x, init_filters * 4, kernel_size=kernel_size, strides=2)
    x = Conv2DNormLReLU(x, init_filters * 4, kernel_size=kernel_size)

    x = Conv2DNormLReLU(x, init_filters * 4, kernel_size=kernel_size)
    x = InvertedRes_block(x, expansion_ratio=2, output_dim=init_filters * 8, stride=1)
    x = InvertedRes_block(x, expansion_ratio=2, output_dim=init_filters * 8, stride=1)
    x = InvertedRes_block(x, expansion_ratio=2, output_dim=init_filters * 8, stride=1)
    x = InvertedRes_block(x, expansion_ratio=2, output_dim=init_filters * 8, stride=1)
    x = Conv2DNormLReLU(x, init_filters * 4)

    x = Unsample(x, init_filters * 4)
    x = Conv2DNormLReLU(x, init_filters * 4, kernel_size=kernel_size)

    x = Unsample(x, init_filters * 2)
    x = Conv2DNormLReLU(x, init_filters * 2, kernel_size=kernel_size)
    x = Conv2DNormLReLU(x, init_filters, kernel_size=7)
    x = Conv2D(x, filters=3, kernel_size=1, strides=1, activation='tanh')

    return tf.keras.models.Model(inputs=input1, outputs=x)


def conv(x, channels, kernel=4, stride=2, use_bias=True):
    return tf.keras.layers.Conv2D(filters=channels,
                                  kernel_size=kernel,
                                  strides=stride,
                                  padding='same',
                                  use_bias=use_bias)(x)


def discriminator(image_size=(64, 64), channels=3, kernel_size=3, init_filters=64, n_dis=3):
    input1 = tf.keras.layers.Input((image_size[0], image_size[1], channels))
    x = input1
    init_filters = init_filters // 2

    x = conv(x, init_filters, kernel=kernel_size, stride=1, use_bias=False)
    x = lrelu(x, 0.2)

    for i in range(1, n_dis):
        x = conv(x, init_filters * 2, kernel=kernel_size, stride=2, use_bias=False)
        x = lrelu(x, 0.2)

        x = conv(x, init_filters * 4, kernel=kernel_size, stride=1, use_bias=False)
        x = layer_norm(x)
        x = lrelu(x, 0.2)

        init_filters = init_filters * 2

    x = conv(x, init_filters * 2, kernel=kernel_size, stride=1, use_bias=False)
    x = layer_norm(x)
    x = lrelu(x, 0.2)

    x = conv(x, channels=1, kernel=kernel_size, stride=1, use_bias=False)

    return tf.keras.models.Model(inputs=input1, outputs=x)

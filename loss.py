import tensorflow as tf


def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss


def Huber_loss(x, y):
    return tf.compat.v1.losses.huber_loss(x, y)


def discriminator_loss(fake_logit,
                       anime_logit,
                       anime_gray_logit,
                       anime_smooth_logit,
                       other_anime_logit=None,
                       other_anime_gray_logit=None,
                       other_anime_smooth_logit=None):
    fake_loss = tf.reduce_mean(tf.square(fake_logit))
    anime_loss = tf.reduce_mean(tf.square(anime_logit - 1.0))
    anime_gray_loss = tf.reduce_mean(tf.square(anime_gray_logit))
    anime_smooth_loss = tf.reduce_mean(tf.square(anime_smooth_logit))
    other_anime_loss = tf.reduce_mean(tf.square(other_anime_logit)) if other_anime_logit is not None else 0.0
    other_anime_gray_loss = tf.reduce_mean(
        tf.square(other_anime_gray_logit)) if other_anime_gray_logit is not None else 0.0
    other_anime_smooth_loss = tf.reduce_mean(
        tf.square(other_anime_smooth_logit)) if other_anime_smooth_logit is not None else 0.0

    # for Hayao : 1.2, 1.2, 1.2, 0.8
    # for Paprika : 1.0, 1.0, 1.0, 0.005
    # for Shinkai: 1.7, 1.7, 1.7, 1.0
    loss = 1.7 * fake_loss \
           + 1.7 * anime_loss \
           + 1.7 * anime_gray_loss \
           + 1.0 * anime_smooth_loss \
           + 1.0 * other_anime_loss \
           + 1.0 * other_anime_gray_loss \
           + 1.0 * other_anime_smooth_loss

    return loss


def generator_loss(fake):
    return tf.reduce_mean(tf.square(fake - 1.0))


def gram(x):
    shape_x = tf.shape(x)
    b = shape_x[0]
    c = shape_x[3]
    x = tf.reshape(x, [b, -1, c])
    return tf.matmul(tf.transpose(x, [0, 2, 1]), x) / tf.cast((tf.size(x) // b), tf.float32)


def con_loss(vgg, real, fake):
    # vgg.build(real)
    real_feature_map = vgg(real)

    # vgg.build(fake)
    fake_feature_map = vgg(fake)

    loss = L1_loss(real_feature_map, fake_feature_map)

    return loss


def style_loss(style, fake):
    return L1_loss(gram(style), gram(fake))


def con_sty_loss(vgg, real, anime_gray, fake):
    # vgg.build(real)
    real_feature_map = vgg(real)

    # vgg.build(fake)
    fake_feature_map = vgg(fake)

    # vgg.build(anime[:fake_feature_map.shape[0]])
    anime_feature_map = vgg(anime_gray)

    c_loss = L1_loss(real_feature_map, fake_feature_map)
    s_loss = style_loss(anime_feature_map, fake_feature_map)

    return c_loss, s_loss


def color_loss(con, fake):
    con = rgb2yuv(con)
    fake = rgb2yuv(fake)

    return L1_loss(con[:, :, :, 0], fake[:, :, :, 0]) + Huber_loss(con[:, :, :, 1], fake[:, :, :, 1]) + Huber_loss(
        con[:, :, :, 2], fake[:, :, :, 2])


def total_variation_loss(inputs):
    """
    A smooth loss in fact. Like the smooth prior in MRF.
    V(y) = || y_{n+1} - y_n ||_2
    """
    dh = inputs[:, :-1, ...] - inputs[:, 1:, ...]
    dw = inputs[:, :, :-1, ...] - inputs[:, :, 1:, ...]
    size_dh = tf.size(dh, out_type=tf.float32)
    size_dw = tf.size(dw, out_type=tf.float32)
    return tf.nn.l2_loss(dh) / size_dh + tf.nn.l2_loss(dw) / size_dw


def rgb2yuv(rgb):
    """
    Convert RGB image into YUV https://en.wikipedia.org/wiki/YUV
    """
    rgb = (rgb + 1.0) / 2.0
    # rgb2yuv_filter = tf.constant([[[[0.299, -0.169, 0.499],
    #                                 [0.587, -0.331, -0.418],
    #                                 [0.114, 0.499, -0.0813]]]])
    # rgb2yuv_bias = tf.constant([0., 0.5, 0.5])
    # temp = tf.nn.conv2d(rgb, rgb2yuv_filter, [1, 1, 1, 1], 'SAME')
    # temp = tf.nn.bias_add(temp, rgb2yuv_bias)
    # return temp
    return tf.image.rgb_to_yuv(rgb)

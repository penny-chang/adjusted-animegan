import tensorflow as tf


@tf.function
def load_image(path):
    img = tf.io.read_file(path)
    return img


@tf.function
def decode_image(img):
    img = tf.io.decode_image(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = img / 127.5 - 1.0  # norm [0, 255] => [-1, 1]
    return img


@tf.function
def random_crop(img, crop_size, with_gray=False):
    img_size = tf.shape(img)

    if crop_size is not None and img_size[0] > crop_size[0] and img_size[1] > crop_size[1]:
        # random resize
        min_size = tf.cast(tf.math.reduce_min([img_size[0], img_size[1]]), tf.float32)
        scale = tf.random.uniform([], minval=crop_size, maxval=min_size, dtype=tf.float32) / min_size  # [1, 0)
        new_size = tf.cast(tf.cast(img_size[0:2], tf.float32) * scale, tf.int32)
        img = tf.image.resize(img, new_size)

        # random_crop
        img = tf.image.random_crop(img, (crop_size[0], crop_size[1], 3))

        # random_flip_left_right
        img = tf.image.random_flip_left_right(img)
    if with_gray:
        return img, rgb_to_grayscale(img)
    return img


def pad_image_to_tile_multiple(img, tile_size, padding="CONSTANT"):
    imagesize = tf.shape(img)[0:2]
    padding_ = tf.cast(tf.math.ceil(imagesize / tile_size), tf.int32) * tile_size - imagesize
    return tf.pad(img, [[0, padding_[0]], [0, padding_[1]], [0, 0]], padding)


def split_image(image3, tile_size):
    image_shape = tf.shape(image3)
    tile_rows = tf.reshape(image3, [image_shape[0], -1, tile_size[1], image_shape[2]])
    serial_tiles = tf.transpose(tile_rows, [1, 0, 2, 3])
    return tf.reshape(serial_tiles, [-1, tile_size[1], tile_size[0], image_shape[2]])


@tf.function
def unsplit_image(tiles, image_shape):
    tile_width = tf.shape(tiles)[1]
    serialized_tiles = tf.reshape(tiles, [-1, image_shape[0], tile_width, image_shape[2]])
    rowwise_tiles = tf.transpose(serialized_tiles, [1, 0, 2, 3])
    return tf.reshape(rowwise_tiles, [image_shape[0], image_shape[1], image_shape[2]])


@tf.function
def pad_to_batch_size(batch, batch_size, mode="CONSTANT"):
    b_size = tf.shape(batch)[0]
    if b_size >= batch_size:
        return batch
    return tf.pad(batch, [[0, batch_size - b_size], [0, 0], [0, 0], [0, 0]], mode=mode)


@tf.function
def write_image(path, image):
    image = (image + 1.0) * 127.5  # norm [-1, 1] => [0, 255]
    image = tf.cast(image, tf.uint8)
    image = tf.io.encode_jpeg(image, format="rgb")
    tf.io.write_file(path, image)


@tf.function
def rgb_to_grayscale(img):
    img = tf.image.rgb_to_grayscale(img)
    img = tf.concat([img, img, img], axis=-1)
    return img


def from_folder(folder,
                image_size=(64, 64),
                shuffle=True,
                seed=1234,
                shuffle_buffer_size=1024,
                batch_size=None,
                with_gray=False,
                repeat=False,
                cache_filename=""):
    ds = tf.data.Dataset.list_files(folder + '*/*.jpg', shuffle=False, name='real')
    num_parallel_calls = tf.data.experimental.AUTOTUNE
    # num_parallel_calls = 20
    ds = ds.map(load_image, num_parallel_calls=num_parallel_calls)

    ds = ds.cache()
    if repeat:
        ds = ds.repeat()

    ds = ds.map(decode_image, num_parallel_calls=num_parallel_calls)

    ds = ds.map(lambda img: random_crop(img, crop_size=image_size, with_gray=with_gray),
                num_parallel_calls=num_parallel_calls)

    if shuffle:
        ds = ds.shuffle(shuffle_buffer_size, seed=seed, reshuffle_each_iteration=True)

    if batch_size is not None:
        ds = ds.batch(batch_size, drop_remainder=True, num_parallel_calls=num_parallel_calls)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

import glob
import os
from pathlib import Path

from common import get_batch_size_and_init_filters, create_dir
from dataset import load_image, write_image, pad_image_to_tile_multiple, pad_to_batch_size, unsplit_image, \
    split_image, decode_image
from model import generator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8
import tensorflow as tf
import numpy as np


def get_gen(output_dir, init_filters, required=False, verbose=2):
    gen = generator(image_size=(None, None), init_filters=init_filters)
    if verbose > 1:
        print(gen.summary())
    ckpt = tf.train.Checkpoint(gen=gen)
    latest_checkpoint = tf.train.latest_checkpoint(output_dir)
    if verbose > 0:
        print(output_dir)
    if latest_checkpoint:
        ckpt.restore(latest_checkpoint).expect_partial()
    elif not required:
        print("Initializing from scratch.")
    else:
        raise RuntimeError('checkpoint not found')
    return gen


def transform_img(batch_size, file, tile_size, transformed_dir, gen):
    print(file)
    filepath = Path(file)
    img = load_image(file)
    img = decode_image(img)
    pad_img = None
    if tile_size is not None:
        pad_img = pad_image_to_tile_multiple(img, tile_size=tile_size)
        tiles = split_image(pad_img, tile_size=tile_size)
    else:
        tiles = [img]
    tiles_ds = tf.data.Dataset.from_tensor_slices(list(tiles))
    tiles_ds = tiles_ds.batch(batch_size, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    tiles_ds = tiles_ds.map(lambda x: pad_to_batch_size(x, batch_size=batch_size),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    tiles_ds = tiles_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    gen_img = None
    if tile_size is not None:
        gen_tiles = np.zeros(shape=(tf.cast(tf.math.ceil(tiles.shape[0] / batch_size) * batch_size, tf.int32),
                                    tiles.shape[1], tiles.shape[2], tiles.shape[3]))
        for i, ts in enumerate(tiles_ds):
            # result = gen.predict(ts)
            result = gen(ts)
            gen_tiles[i * batch_size:i * batch_size + batch_size] = result

        gen_tiles = gen_tiles[:tiles.shape[0]]

        gen_img = unsplit_image(gen_tiles, image_shape=tf.shape(pad_img))
        gen_img = gen_img[0:tf.shape(img)[0], 0:tf.shape(img)[1], :]
    else:
        for ts in tiles_ds:
            gen_img = ts[0]
            break
    create_dir(transformed_dir)
    transformed_img_path = transformed_dir + '/' + filepath.name
    write_image(transformed_img_path, gen_img)
    return transformed_img_path


def test(batch_size, output_dir, input_folder, init_filters, tile_size=(64, 64)):
    gen = get_gen(output_dir, init_filters)
    transformed_dir = output_dir + '/transformed/'

    for file in glob.glob(input_folder + '/**/*.jpg', recursive=True):
        transform_img(batch_size, file, tile_size, transformed_dir, gen)


if __name__ == "__main__":
    input_folder = './dataset/test/test'
    result_folder_name = '[COCO_10000][ghibli_pics_500][conan_wallpaper][ep-500][i_ep-20][batch-16][size-64][f-16][i_lr-0.0002][g_lr-2e-05][d_lr-4e-05]'
    result_root_path = './result'
    batch_size, init_filters = get_batch_size_and_init_filters(result_folder_name)
    output_dir = result_root_path + '/' + result_folder_name

    print("Tensorflow Version", tf.__version__)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    test(batch_size=batch_size,
         output_dir=output_dir,
         input_folder=input_folder,
         init_filters=init_filters,
         tile_size=(512, 512))

import os

import matplotlib.pyplot as plt

from common import get_result_folder_name
from dataset import from_folder
from loss import con_loss, con_sty_loss, total_variation_loss, color_loss, generator_loss, discriminator_loss
from model import generator, discriminator
from vgg19 import Vgg19

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8
import tensorflow as tf
import numpy as np
import time


@tf.function
def init_step(
        real_img,
        vgg: Vgg19,
        gen,
        init_opt,
        con_weight=1.2,  # 1.5 for Hayao, 2.0 for Paprika, 1.2 for Shinkai
):
    with tf.GradientTape() as init_tape:
        gen_img = gen(real_img, training=True)

        # init pharse
        init_c_loss = con_loss(vgg, real_img, gen_img)
        init_loss = con_weight * init_c_loss

    gradients_of_init = init_tape.gradient(init_loss, gen.trainable_variables)

    init_opt.apply_gradients(zip(gradients_of_init, gen.trainable_variables))
    return init_loss


@tf.function
def train_step(
    real_img,
    anime_img,
    anime_gray_img,
    anime_smooth_img,
    vgg: Vgg19,
    gen,
    disc,
    gen_opt,
    disc_opt,
    g_adv_weight=300.0,  # Weight about GAN
    d_adv_weight=300.0,  # Weight about GAN
    con_weight=1.2,  # 1.5 for Hayao, 2.0 for Paprika, 1.2 for Shinkai
    sty_weight=2.0,  # 2.5 for Hayao, 0.6 for Paprika, 2.0 for Shinkai
    color_weight=10.0,  # 15. for Hayao, 50. for Paprika, 10. for Shinkai
    tv_weight=1.0,  # 1. for Hayao, 0.1 for Paprika, 1. for Shinkai
    other_anime_img=None,
    other_anime_gray_img=None,
    other_anime_smooth_img=None,
):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_img = gen(real_img, training=True)

        fake_logit = disc(fake_img)
        anime_logit = disc(anime_img)
        anime_gray_logit = disc(anime_gray_img)
        anime_smooth_logit = disc(anime_smooth_img)
        other_anime_logit = disc(other_anime_img) if other_anime_img is not None else None
        other_anime_gray_logit = disc(other_anime_gray_img) if other_anime_gray_img is not None else None
        other_anime_smooth_logit = disc(other_anime_smooth_img) if other_anime_smooth_img is not None else None

        # gan
        # c_loss: content loss, s_loss: style loss
        c_loss, s_loss = con_sty_loss(vgg, real_img, anime_gray_img, fake_img)
        tv_loss = tv_weight * total_variation_loss(fake_img)
        t_loss = con_weight * c_loss + sty_weight * s_loss + color_loss(real_img, fake_img) * color_weight + tv_loss

        g_loss = g_adv_weight * generator_loss(fake_logit)
        d_loss = d_adv_weight * discriminator_loss(fake_logit=fake_logit,
                                                   anime_logit=anime_logit,
                                                   anime_gray_logit=anime_gray_logit,
                                                   anime_smooth_logit=anime_smooth_logit,
                                                   other_anime_logit=other_anime_logit,
                                                   other_anime_gray_logit=other_anime_gray_logit,
                                                   other_anime_smooth_logit=other_anime_smooth_logit)

        gen_loss = t_loss + g_loss
        disc_loss_total = d_loss

    gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss_total, disc.trainable_variables)

    gen_opt.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    disc_opt.apply_gradients(zip(gradients_of_discriminator, disc.trainable_variables))
    return gen_loss, disc_loss_total


def train(real_dataset_name, anime_dataset_name, other_anime_dataset_name, epochs, batch_size, output_dir, image_size,
          init_lr, gen_lr, disc_lr, init_epochs, init_filters):
    tf.keras.backend.clear_session()
    real_ds = from_folder(f'./dataset/{real_dataset_name}',
                          image_size=image_size,
                          batch_size=batch_size,
                          shuffle_buffer_size=batch_size * 10,
                          cache_filename=f"{output_dir}/real.cache")
    anime_ds = from_folder(
        f'./dataset/{anime_dataset_name}/style',
        image_size=image_size,
        batch_size=batch_size,
        shuffle_buffer_size=batch_size * 10,
        cache_filename=f"{output_dir}/anime.cache",
        with_gray=True,
        repeat=True,  # repeat to fit number of real images
    )
    anime_smooth_ds = from_folder(
        f'./dataset/{anime_dataset_name}/smooth',
        image_size=image_size,
        batch_size=batch_size,
        shuffle_buffer_size=batch_size * 10,
        cache_filename=f"{output_dir}/anime_smooth.cache",
        repeat=True,  # repeat to fit number of real images
    )
    other_anime_ds = from_folder(
        f'./dataset/{other_anime_dataset_name}/style',
        image_size=image_size,
        batch_size=batch_size,
        shuffle_buffer_size=batch_size * 10,
        cache_filename=f"{output_dir}/other_anime.cache",
        with_gray=True,
        repeat=True,  # repeat to fit number of real images
    ) if other_anime_dataset_name is not None else None
    other_anime_smooth_ds = from_folder(
        f'./dataset/{other_anime_dataset_name}/smooth',
        image_size=image_size,
        batch_size=batch_size,
        shuffle_buffer_size=batch_size * 10,
        cache_filename=f"{output_dir}/other_anime_smooth.cache",
        repeat=True,  # repeat to fit number of real images
    ) if other_anime_dataset_name is not None else None
    test_ds = from_folder('./dataset/test/real', image_size=None, batch_size=batch_size, shuffle=False)

    vgg19 = Vgg19()
    vgg19.build(image_size=image_size)
    vgg = vgg19.get_conv4_4_no_activation_model()
    gen = generator(image_size=(None, None), init_filters=init_filters)
    disc = discriminator(image_size=image_size, init_filters=init_filters)

    print(gen.summary())
    print(disc.summary())
    init_opt = tf.keras.optimizers.Adam(learning_rate=init_lr, beta_1=0.5)
    gen_opt = tf.keras.optimizers.Adam(learning_rate=gen_lr, beta_1=0.5)
    disc_opt = tf.keras.optimizers.Adam(learning_rate=disc_lr, beta_1=0.5)

    ckpt = tf.train.Checkpoint(disc=disc, gen=gen, disc_opt=disc_opt, gen_opt=gen_opt)
    manager = tf.train.CheckpointManager(ckpt, output_dir, max_to_keep=10, step_counter=100)

    init_ep = 1
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        ckpt.restore(manager.latest_checkpoint)
        init_ep = int(manager.latest_checkpoint[manager.latest_checkpoint.rindex("-") + 1:])
    else:
        print("Initializing from scratch.")

        with open(f"{output_dir}/history.csv", "w") as csv_file:
            csv_file.write("Train Time,Save Time,Plot Time,Init Loss,Gen Loss,Disc Loss\n")

    plot(0, gen, output_dir, test_ds, plot_origin=True)
    plot(0, gen, output_dir, test_ds)
    init_loss = 0.0
    gen_loss = 0.0
    disc_loss = 0.0
    for ep in range(init_ep, epochs + 1):

        print('Epoch: %d of %d' % (ep, epochs))
        start_time = time.time()

        if ep <= init_epochs:
            init_loss = []
            for real_img in real_ds:
                i_loss = init_step(real_img=real_img, vgg=vgg, gen=gen, init_opt=init_opt)
                init_loss.append(i_loss)
            if ep == 1:
                print(f"Total Batch: {len(init_loss)}")
            init_loss = np.mean(np.asarray(init_loss))

        else:
            gen_loss = []
            disc_loss = []
            if other_anime_ds is not None:
                for real_img, anime_img, anime_smooth, other_anime, other_anime_smooth in zip(
                        real_ds, anime_ds, anime_smooth_ds, other_anime_ds, other_anime_smooth_ds):
                    g_loss, d_loss = train_step(real_img=real_img,
                                                anime_img=anime_img[0],
                                                anime_gray_img=anime_img[1],
                                                anime_smooth_img=anime_smooth,
                                                vgg=vgg,
                                                gen=gen,
                                                disc=disc,
                                                gen_opt=gen_opt,
                                                disc_opt=disc_opt,
                                                other_anime_img=other_anime[0],
                                                other_anime_gray_img=other_anime[1],
                                                other_anime_smooth_img=other_anime_smooth)
                    gen_loss.append(g_loss)
                    disc_loss.append(d_loss)
            else:
                for real_img, anime_img, anime_smooth in zip(real_ds, anime_ds, anime_smooth_ds):
                    g_loss, d_loss = train_step(real_img=real_img,
                                                anime_img=anime_img[0],
                                                anime_gray_img=anime_img[1],
                                                anime_smooth_img=anime_smooth,
                                                vgg=vgg,
                                                gen=gen,
                                                disc=disc,
                                                gen_opt=gen_opt,
                                                disc_opt=disc_opt)
                    gen_loss.append(g_loss)
                    disc_loss.append(d_loss)
            gen_loss = np.mean(np.asarray(gen_loss))
            disc_loss = np.mean(np.asarray(disc_loss))

        save_start_time = time.time()

        if ep <= init_epochs or ep % 10 == 0:
            manager.save(checkpoint_number=ep)

        plot_start_time = time.time()

        if ep <= init_epochs or ep % 10 == 0:
            plot(ep, gen, output_dir, test_ds)

        end_time = time.time()
        message = f"{end_time - start_time :.3f}s"
        message += f" - init_loss: {init_loss:.3f}"
        message += f" - gen_loss: {gen_loss:.3f}"
        message += f" - disc_loss: {disc_loss:.3f}"
        with open(f"{output_dir}/history.csv", "a") as csv_file:
            csv_file.write(
                f"{save_start_time - start_time},{plot_start_time - save_start_time},{end_time - plot_start_time},{init_loss},{gen_loss},{disc_loss}\n"
            )
        print(message)


def plot(ep, gen, output_dir, test_ds, plot_origin=False):
    fig = plt.figure(figsize=(20, 20))
    gen_test_img = np.array(list(test_ds.take(1))[0])
    if not plot_origin:
        gen_test_img = gen(gen_test_img, training=False)

    for i in range(gen_test_img.shape[0]):
        plt.subplot(4, 4, i + 1)
        image = gen_test_img[i, :, :, :] * 0.5 + 0.5
        plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    if plot_origin:
        plt.savefig(output_dir + '/image.png')
    else:
        plt.savefig(output_dir + '/image_at_epoch_{:03d}.png'.format(ep))
    fig.clf()
    plt.close()


def create_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


if __name__ == "__main__":
    real_dataset_name = 'train2014'
    anime_dataset_name = 'ghibli_pics'
    other_anime_dataset_name = None
    epochs = 500
    init_epochs = 20
    batch_size = 16
    image_size = (64, 64)
    init_filters = 16
    init_lr = 2e-4
    gen_lr = 2e-5
    disc_lr = 4e-5
    result_root_path = './result'
    result_folder_name = get_result_folder_name(real_dataset_name=real_dataset_name,
                                                anime_dataset_name=anime_dataset_name,
                                                other_anime_dataset_name=other_anime_dataset_name,
                                                epochs=epochs,
                                                init_epochs=init_epochs,
                                                batch_size=batch_size,
                                                image_size=image_size,
                                                init_filters=init_filters,
                                                init_lr=init_lr,
                                                gen_lr=gen_lr,
                                                disc_lr=disc_lr)
    output_dir = result_root_path + '/' + result_folder_name
    create_dir(output_dir)

    print("Tensorflow Version", tf.__version__)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    train(real_dataset_name=real_dataset_name,
          anime_dataset_name=anime_dataset_name,
          other_anime_dataset_name=other_anime_dataset_name,
          epochs=epochs,
          batch_size=batch_size,
          output_dir=output_dir,
          image_size=image_size,
          init_lr=init_lr,
          gen_lr=gen_lr,
          disc_lr=disc_lr,
          init_epochs=init_epochs,
          init_filters=init_filters)
    # make_gif(out_path)

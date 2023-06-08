import os
import re


def create_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def get_result_folder_name(real_dataset_name, anime_dataset_name, other_anime_dataset_name, epochs, init_epochs,
                           batch_size, image_size, init_filters, init_lr, gen_lr, disc_lr):
    result_folder_name = f'[{real_dataset_name}]' \
                         f'[{anime_dataset_name}]'

    if other_anime_dataset_name is not None:
        result_folder_name += f"[{other_anime_dataset_name}]"

    result_folder_name += f'[ep-{epochs}]' \
                          f'[i_ep-{init_epochs}]' \
                          f'[batch-{batch_size}]' \
                          f'[size-{image_size[0]}]' \
                          f'[f-{init_filters}]' \
                          f'[i_lr-{init_lr}]' \
                          f'[g_lr-{gen_lr}]' \
                          f'[d_lr-{disc_lr}]'

    return result_folder_name


def get_batch_size_and_init_filters(result_folder_name):
    match = re.search(r'.*batch-(?P<batch_size>[^]]+).*f-(?P<init_filters>[^]]+).*', result_folder_name)
    return int(match.group('batch_size')), int(match.group('init_filters'))

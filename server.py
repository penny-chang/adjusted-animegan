import os
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from common import create_dir
from test import get_gen, transform_img

static_dirname = 'static'
upload_dirname = 'upload'
transformed_dirname = "transformed"
app = Flask(__name__, template_folder='templates', static_folder=static_dirname)

app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, upload_dirname)
app.config['TRANSFORMED_FOLDER'] = os.path.join(app.static_folder, transformed_dirname)

result_folder_name_dict = {
    'Result':
    '[COCO_10000][ghibli_pics_500][ep-500][i_ep-20][batch-16][size-64][f-16][i_lr-0.0002][g_lr-2e-05][d_lr-4e-05]',
}

gen_cache = dict()


def test_for_api(model, img_path, transformed_dir):
    batch_size = 16
    gen = gen_cache[model]
    transformed_img_path = transform_img(batch_size, img_path, (256, 256), transformed_dir, gen)
    return transformed_img_path


@app.route('/', methods=["GET"])
def index():
    history_list = [{
        'filename': path.name,
        'last_modified': datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y/%m/%d %H:%M:%S'),
    } for path in sorted(
        Path(os.path.join(app.config['TRANSFORMED_FOLDER'], 'Result')).rglob("*.*"), key=lambda path: path.stat().st_mtime, reverse=True)]
    result_list = result_folder_name_dict.keys()
    print(f'history_list={history_list}')
    return render_template(
        'index.html',
        data=result_list,
        history=history_list,  # comment to disable
    )


@app.route('/result', methods=["GET"])
def result():
    return render_template('result.html', **request.args, models=result_folder_name_dict.keys())


@app.route('/transform', methods=["POST"])
def upload_file():
    if request.method == 'POST':
        img_file = request.files['image']
        img_filename = secure_filename(img_file.filename)

        # save file
        create_dir(os.path.join(app.config['UPLOAD_FOLDER']))
        img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)

        # transform
        for model in result_folder_name_dict.keys():
            test_for_api(model, img_path, transformed_dir=os.path.join(app.config['TRANSFORMED_FOLDER'], model))

        # redirect to result page
        return redirect(url_for('result', filename=img_filename))


if __name__ == '__main__':
    print('Model Initializing')
    for model, result_folder_name in result_folder_name_dict.items():
        gen = get_gen('./result/' + result_folder_name, 16, required=True, verbose=1)
        gen.predict(np.zeros((16, 32, 32, 3)))
        gen_cache[model] = gen
    print('API Initializing')
    app.run(host="localhost", port=5000, debug=True)

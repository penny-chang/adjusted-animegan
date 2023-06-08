# An adjusted AnimeGAN

This is a demo project for NN course at NTNU in 2023.

## Run the demo site

1. Install the requirements.

    `pip install -r requirements.txt`

2. Run the server.

    `python server.py`

3. The website url will be shown in the console.

4. You may change the server code in `server.py` for it to load other model if you want.

## Train your own model

1. Download the real-world image dataset from [COCO website](https://cocodataset.org/#home). Put the images in `/dataset/train2014` folder.

2. Download the anime images from [Studio Ghibli website](https://www.ghibli.jp/works). Put the images in `/dataset/ghibli_pics` folder.

3. If you wish to load the file or save the results in your desired path, change the code in `train.py` file.

4. Install the requirements and run `train.py`.

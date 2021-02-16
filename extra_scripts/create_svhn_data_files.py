import argparse
import os

import scipy.io
from tqdm import tqdm
from PIL import Image
from torchvision.datasets.utils import download_url


def download_dataset(root: str):
    """
    Download the CLEVR dataset archive and expand it in the folder provided as parameter
    """
    TRAIN_URL = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
    TEST_URL = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
    download_url(url=TRAIN_URL, root=root)
    download_url(url=TEST_URL, root=root)


def create_svnh_disk_folder_split(dataset, folder: str):
    for label in range(1, 11):
        os.makedirs(os.path.join(folder, f"digit_{label}"), exist_ok=True)

    images = dataset['X']
    targets = dataset['y']
    nb_sample = targets.shape[0]
    with tqdm(total=nb_sample) as progress_bar:
        for i in range(nb_sample):
            image = Image.fromarray(images[:, :, :, i])
            label = f"digit_{targets[i][0]}"
            image.save(os.path.join(folder, label, f"image_{i}.jpg"))
            progress_bar.update(1)


def create_svnh_disk_folder(root: str):
    create_svnh_disk_folder_split(
        dataset=scipy.io.loadmat(os.path.join(root, "train_32x32.mat")),
        folder=os.path.join(root, "train")
    )
    create_svnh_disk_folder_split(
        dataset=scipy.io.loadmat(os.path.join(root, "test_32x32.mat")),
        folder=os.path.join(root, "test")
    )


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str, help='Where the classification dataset will be written')
    return parser


if __name__ == '__main__':
    args = get_argument_parser().parse_args()
    download_dataset(args.output)
    create_svnh_disk_folder(args.output)

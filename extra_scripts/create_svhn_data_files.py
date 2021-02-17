import argparse
import os

import scipy.io
from tqdm import tqdm
from PIL import Image
from torchvision.datasets.utils import download_url


def get_argument_parser():
    """
    List of arguments supported by the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="The input folder contains the SVHN data files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The output folder containing the disk_folder output",
    )
    parser.add_argument(
        "-d",
        "--download",
        action="store_const",
        const=True,
        default=False,
        help="To download the original dataset and decompress it in the input folder",
    )
    return parser


def download_dataset(root: str):
    """
    Download the CLEVR dataset archive and expand it in the folder provided as parameter
    """
    TRAIN_URL = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
    TEST_URL = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
    download_url(url=TRAIN_URL, root=root)
    download_url(url=TEST_URL, root=root)


def create_svnh_disk_folder_split(dataset, folder: str):
    """
    Create one split of the SVHN dataset (ex: "train" or "test")
    """
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


def create_svnh_disk_folder(input_path: str, output_path: str):
    create_svnh_disk_folder_split(
        dataset=scipy.io.loadmat(os.path.join(input_path, "train_32x32.mat")),
        folder=os.path.join(output_path, "train")
    )
    create_svnh_disk_folder_split(
        dataset=scipy.io.loadmat(os.path.join(input_path, "test_32x32.mat")),
        folder=os.path.join(output_path, "test")
    )


if __name__ == '__main__':
    args = get_argument_parser().parse_args()
    if args.download:
        download_dataset(args.input)
    create_svnh_disk_folder(input_path=args.input, output_path=args.output)

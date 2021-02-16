import argparse
import os
import shutil
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm


def get_argument_parser():
    """
    List of arguments supported by the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        help="Path to the folder containing expanded EuroSAT.zip archive")
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        help="Folder where the classification dataset will be written")
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
    Download the EuroSAT dataset archive and expand it in the folder provided as parameter
    """
    URL = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
    download_and_extract_archive(url=URL, download_root=root)


class _EuroSAT(VisionDataset):
    """
    Dataset used to parallelize the transformation of the dataset via a DataLoader
    """

    IMAGE_FOLDER = "2750"
    TRAIN_SAMPLES = 1000
    VALID_SAMPLES = 500

    def __init__(self, root: str, train: bool, transform=None, target_transform=None):
        super().__init__(root=root, transform=transform, target_transform=target_transform)
        self.train = train
        self.image_folder = os.path.join(root, self.IMAGE_FOLDER)
        self.images = []
        self.targets = []
        self.labels = list(sorted(os.listdir(self.image_folder)))
        for i, label in enumerate(self.labels):
            label_path = os.path.join(self.image_folder, label)
            files = list(sorted(os.listdir(label_path)))
            if train:
                self.images.extend(files[:self.TRAIN_SAMPLES])
                self.targets.extend([i] * self.TRAIN_SAMPLES)
            else:
                self.images.extend(files[self.TRAIN_SAMPLES:self.TRAIN_SAMPLES + self.VALID_SAMPLES])
                self.targets.extend([i] * self.VALID_SAMPLES)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        image_name = self.images[idx]
        target = self.labels[self.targets[idx]]
        image_path = os.path.join(self.image_folder, target, image_name)
        return image_name, image_path, target


def create_disk_folder_split(dataset: _EuroSAT, split_path: str):
    for label in dataset.labels:
        os.makedirs(os.path.join(split_path, label), exist_ok=True)
    loader = DataLoader(dataset, num_workers=8, batch_size=1, collate_fn=lambda x: x[0])
    for image_name, image_path, target in tqdm(loader):
        shutil.copy(image_path, os.path.join(split_path, target, image_name))


def create_euro_sat_disk_folder(input_path: str, output_path: str):
    print("Creating the training split...")
    create_disk_folder_split(
        dataset=_EuroSAT(input_path, train=True),
        split_path=os.path.join(output_path, "train")
    )
    print("Creating the validation split...")
    create_disk_folder_split(
        dataset=_EuroSAT(input_path, train=False),
        split_path=os.path.join(output_path, "val")
    )


if __name__ == '__main__':
    """
    Example usage:

    ```
    python extra_scripts/create_ucf101_data_files.py -i /path/to/euro_sat -o /output_path/to/euro_sat -d
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_dataset(args.input)
    create_euro_sat_disk_folder(args.input, args.output)

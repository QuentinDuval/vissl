import argparse
import os
from collections import OrderedDict
from typing import NamedTuple, Tuple

import h5py
from PIL import Image
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
        "-i",
        "--input",
        type=str,
        help="The input folder contains the Patch Camelyon data files",
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
    parser.add_argument(
        '-w',
        '--workers',
        type=int,
        default=8,
        help="Number of parallel workers")
    return parser


class Split(NamedTuple):
    file: str
    url: str


class PatchCamelyon(VisionDataset):
    """
    Dataset used to parallelize the transformation of the dataset via a DataLoader
    """

    _FILES = OrderedDict(
        {
            "train_x": Split(
                "camelyonpatch_level_2_split_train_x.h5",
                "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_x.h5.gz",
            ),
            "train_y": Split(
                "camelyonpatch_level_2_split_train_y.h5",
                "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_y.h5.gz",
            ),
            "valid_x": Split(
                "camelyonpatch_level_2_split_valid_x.h5",
                "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_x.h5.gz",
            ),
            "valid_y": Split(
                "camelyonpatch_level_2_split_valid_y.h5",
                "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_y.h5.gz",
            ),
        }
    )

    def __init__(
        self, root: str, split: str, transform=None, target_transform=None, download: bool = False
    ):
        super().__init__(
            root=root, transform=transform, target_transform=target_transform
        )

        # Checking the presence of the data and downloading it if not present
        for file, url in self._FILES.values():
            if not os.path.exists(os.path.join(root, file)):
                if download:
                    download_and_extract_archive(url=url, download_root=root)
                else:
                    raise ValueError(f"Missing file {file} in {root}")

        # Loading the data
        self.x = self._open_h5_file(self._FILES[f"{split}_x"].file)["x"]
        self.y = self._open_h5_file(self._FILES[f"{split}_y"].file)["y"]

    def _open_h5_file(self, file_name: str):
        return h5py.File(os.path.join(self.root, file_name), "r")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        img = Image.fromarray(self.x[idx])
        label = self.y[idx][0][0]
        return img, label


def to_disk_folder(dataset: PatchCamelyon, output_folder: str, num_workers: int):
    os.makedirs(os.path.join(output_folder, "tumor"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "no_tumor"), exist_ok=True)
    loader = DataLoader(dataset, num_workers=num_workers, batch_size=1, collate_fn=lambda x: x[0])
    for i, (x, y) in enumerate(tqdm(loader)):
        folder = os.path.join(output_folder, "tumor" if y == 1 else "no_tumor")
        with open(os.path.join(folder, f"img_{i}.jpg"), "w") as img_file:
            x.save(img_file)


def create_data_files():
    # Parse command arguments
    parser = get_argument_parser()
    args = parser.parse_args()
    input_folder = os.path.expanduser(args.input)
    output_folder = os.path.expanduser(args.output)
    output_train_folder = os.path.join(output_folder, "train")
    output_valid_folder = os.path.join(output_folder, "val")
    if args.download:
        os.makedirs(input_folder, exist_ok=True)

    # Create the dataset and optionally downloads it
    print(args.download)
    train_set = PatchCamelyon(root=input_folder, split="train", download=args.download)
    valid_set = PatchCamelyon(root=input_folder, split="valid", download=False)

    # Transform the datasets to disk_folder
    to_disk_folder(train_set, output_train_folder, num_workers=args.workers)
    to_disk_folder(valid_set, output_valid_folder, num_workers=args.workers)


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/create_patch_camelyon_data_files.py \
        -i /path/to/pcam \
        -o /output_path/to/pcam \
        -d
    ```
    """
    create_data_files()

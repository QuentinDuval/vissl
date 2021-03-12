# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import numpy as np
import os
import torchvision.datasets as datasets

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
        help="Path to the folder containing the original imagenet-r folder",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Folder where the classification dataset will be written",
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
    Download the Imagenet-R dataset archive and expand it in the folder provided as parameter
    """
    URL = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
    download_and_extract_archive(url=URL, download_root=root)


def create_imagenet_r_test_files(input_path: str, output_path: str):
    """
    Create a test split for ImageNet, based on the ImageNet-R dataset
    """

    # Map the images of Imagenet-R to their corresponding targets in Imagenet
    image_paths = []
    image_labels = []
    target_mapping = get_imagenet_target_mapping()
    imagenet_r = datasets.ImageFolder(root=input_path, loader=lambda x: x)
    for image_path, target_id in tqdm(imagenet_r):
        image_paths.append(image_path)
        image_labels.append(target_mapping[target_id])

    # Save the these lists in the disk_filelist format
    os.makedirs(output_path, exist_ok=True)
    img_info_out_path = os.path.join(output_path, f"test_images.npy")
    label_info_out_path = os.path.join(output_path, f"test_labels.npy")
    np.save(img_info_out_path, np.array(image_paths))
    np.save(label_info_out_path, np.array(image_labels))


def get_imagenet_target_mapping():
    """
    Map the target index of Imagenet-R to the target index of Imagenet
    """
    return {0: 1,
            1: 2,
            2: 4,
            3: 6,
            4: 8,
            5: 9,
            6: 11,
            7: 13,
            8: 22,
            9: 23,
            10: 26,
            11: 29,
            12: 31,
            13: 39,
            14: 47,
            15: 63,
            16: 71,
            17: 76,
            18: 79,
            19: 84,
            20: 90,
            21: 94,
            22: 96,
            23: 97,
            24: 99,
            25: 100,
            26: 105,
            27: 107,
            28: 113,
            29: 122,
            30: 125,
            31: 130,
            32: 132,
            33: 144,
            34: 145,
            35: 147,
            36: 148,
            37: 150,
            38: 151,
            39: 155,
            40: 160,
            41: 161,
            42: 162,
            43: 163,
            44: 171,
            45: 172,
            46: 178,
            47: 187,
            48: 195,
            49: 199,
            50: 203,
            51: 207,
            52: 208,
            53: 219,
            54: 231,
            55: 232,
            56: 234,
            57: 235,
            58: 242,
            59: 245,
            60: 247,
            61: 250,
            62: 251,
            63: 254,
            64: 259,
            65: 260,
            66: 263,
            67: 265,
            68: 267,
            69: 269,
            70: 276,
            71: 277,
            72: 281,
            73: 288,
            74: 289,
            75: 291,
            76: 292,
            77: 293,
            78: 296,
            79: 299,
            80: 301,
            81: 308,
            82: 309,
            83: 310,
            84: 311,
            85: 314,
            86: 315,
            87: 319,
            88: 323,
            89: 327,
            90: 330,
            91: 334,
            92: 335,
            93: 337,
            94: 338,
            95: 340,
            96: 341,
            97: 344,
            98: 347,
            99: 353,
            100: 355,
            101: 361,
            102: 362,
            103: 365,
            104: 366,
            105: 367,
            106: 368,
            107: 372,
            108: 388,
            109: 390,
            110: 393,
            111: 397,
            112: 401,
            113: 407,
            114: 413,
            115: 414,
            116: 425,
            117: 428,
            118: 430,
            119: 435,
            120: 437,
            121: 441,
            122: 447,
            123: 448,
            124: 457,
            125: 462,
            126: 463,
            127: 469,
            128: 470,
            129: 471,
            130: 472,
            131: 476,
            132: 483,
            133: 487,
            134: 515,
            135: 546,
            136: 555,
            137: 558,
            138: 570,
            139: 579,
            140: 583,
            141: 587,
            142: 593,
            143: 594,
            144: 596,
            145: 609,
            146: 613,
            147: 617,
            148: 621,
            149: 629,
            150: 637,
            151: 657,
            152: 658,
            153: 701,
            154: 717,
            155: 724,
            156: 763,
            157: 768,
            158: 774,
            159: 776,
            160: 779,
            161: 780,
            162: 787,
            163: 805,
            164: 812,
            165: 815,
            166: 820,
            167: 824,
            168: 833,
            169: 847,
            170: 852,
            171: 866,
            172: 875,
            173: 883,
            174: 889,
            175: 895,
            176: 907,
            177: 928,
            178: 931,
            179: 932,
            180: 933,
            181: 934,
            182: 936,
            183: 937,
            184: 943,
            185: 945,
            186: 947,
            187: 948,
            188: 949,
            189: 951,
            190: 953,
            191: 954,
            192: 957,
            193: 963,
            194: 965,
            195: 967,
            196: 980,
            197: 981,
            198: 983,
            199: 988}


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/create_imagenet_r_data_files.py -i /path/to/imagenet_r/ -o /output_path/to/imagenet_r -d
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_dataset(args.input)
    input_path = os.path.join(args.input, "imagenet-r")
    create_imagenet_r_test_files(input_path=input_path, output_path=args.output)

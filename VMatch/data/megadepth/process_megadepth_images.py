# Credit to https://github.com/zju3dv/LoFTR/issues/276#issuecomment-1600921374

import os
from PIL import Image
import argparse


def process_megadepth_images(root_directory):
    print(f"Processing images in {root_directory}")
    for folder in os.listdir(root_directory):
        four_digit_directory = os.path.join(root_directory,folder)
        for dense_folder in os.listdir(four_digit_directory):
            image_directory =  os.path.join(four_digit_directory,dense_folder,'imgs')
            for image in os.listdir(image_directory):
                if 'JPG' in image:
                    new_name = image.replace('JPG', 'jpg')
                    old_path = os.path.join(image_directory, image)
                    new_path = os.path.join(image_directory, new_name)
                    os.rename(old_path, new_path)
                if 'png' in image:
                    new_name = image.replace('png', 'jpg')
                    old_path = os.path.join(image_directory, image)
                    new_path = os.path.join(image_directory, new_name)
                    png_img = Image.open(old_path)
                    png_img.save(new_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MegaDepth images")
    parser.add_argument("--root_directory", type=str, help="Root directory of MegaDepth")
    args = parser.parse_args()
    process_megadepth_images(args.root_directory)

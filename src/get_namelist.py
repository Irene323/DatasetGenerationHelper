import argparse
import numpy as np
import pylab as plt
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagefolder", required=True)
    parser.add_argument("--lstrip", required=True)
    parser.add_argument("--rstrip", required=True)
    parser.add_argument("--outtxtfolder", required=True)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    imagefolder = args.imagefolder
    leftstrip = args.lstrip
    rightstrip = args.rstrip
    txtdir = args.outtxtfolder

    with open(txtdir, 'w') as w:
        for _, _, files in os.walk(imagefolder):
            print(files)
            files = sorted(files)
            for file in files:
                print(file)
                w.write(file.lstrip(leftstrip).rstrip(rightstrip)+'\n')


if __name__ == "__main__":
    main()
    # Usage:
    # --imagefolder /cv/projects/boneDescriptors/projects/bones/colmap/Basle/Ailurus_fulgens/dense_2000/images --lstrip IMG_ --rstrip .JPG --outtxtfolder ../out/namelist.txt
    # --imagefolder /cv/projects/boneDescriptors/projects/bones/colmap/Basle/Apteryx_owenii/dense/images --lstrip IMG_ --rstrip .JPG --outtxtfolder ../out/ao/namelist.txt
import argparse
import numpy as np
import pylab as plt
import os


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagefolder", required=True)
    parser.add_argument("--binfolder", required=True)
    parser.add_argument("--txtfolder", required=True)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    imagefolder = args.imagefolder
    binfolder = args.binfolder
    outdir = args.txtfolder

    separator = "/"

    isExists = os.path.exists(outdir)
    if not isExists:
        os.makedirs(outdir)

    for root, dirs, files in os.walk(imagefolder):
        # print(root)
        # print(dirs)
        print(files)
        files = sorted(files)
        for file in files:
            print(file)
            # for every image i.e. every depth map
            depthfilename = file + '.geometric.bin'
            txtname = file.lstrip('IMG_').rstrip('.JPG') + '.txt'
            print(depthfilename)
            print(txtname)
            depthmap = read_array(binfolder + separator + depthfilename)
            print(depthmap.shape)
            txtfile = open(outdir + separator + txtname, 'w')
            for i in range(depthmap.shape[0]):
                for j in range(depthmap.shape[1]):
                    txtfile.write(str(depthmap[i][j]))
                    txtfile.write('\t')
                txtfile.write('\n')
            txtfile.close()


if __name__ == "__main__":
    main()
    # Usage:
    #--imagefolder /cv/projects/boneDescriptors/projects/bones/colmap/Basle/Ailurus_fulgens/dense_2000/images --binfolder /cv/projects/boneDescriptors/projects/bones/colmap/Basle/Ailurus_fulgens/dense_2000/stereo/depth_maps --txtfolder ../out/af/depth_maps
    #--imagefolder /cv/projects/boneDescriptors/projects/bones/colmap/Basle/Apteryx_owenii/dense/images --binfolder /cv/projects/boneDescriptors/projects/bones/colmap/Basle/Apteryx_owenii/dense/stereo/depth_maps --txtfolder ../out/ao/depth_maps
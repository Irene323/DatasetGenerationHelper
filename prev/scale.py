import numpy as np
import argparse
from PIL import Image, ImageFilter
import os
import cv2
import math
import tempfile
import IPython.display
from prev.srgb import to_linear, from_linear
import glob


def show_cv_image(img):
    # Can not use Matplotlib because it scales images again.
    (handle, tmpname) = tempfile.mkstemp(suffix='.png')
    os.close(handle)
    cv2.imwrite(tmpname, img)
    IPython.display.display(IPython.display.Image(tmpname))
    os.unlink(tmpname)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inpath", required=True)
    parser.add_argument("--outpath", required=True)
    args = parser.parse_args()
    return args


def getSigmas():
    ret = []
    s = 3
    sigma = 1.6
    for a in range(4):
        row = []
        for i in range(s + 3):
            row.append((2 ** a) * (2 ** (float(i) / float(s))) * sigma)
        ret.append(row)
    return ret


def getBlur(img, sigma_now, sigma_aim):
    kernel_size = (0, 0)
    fimg = to_linear(img)
    fblur = cv2.GaussianBlur(fimg, kernel_size, (sigma_aim ** 2.0 - sigma_now ** 2.0) ** .5)
    imgblur = from_linear(fblur)
    return imgblur


def getResize(img, fx, fy):
    fimg = to_linear(img)
    fres = cv2.resize(fimg, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    imgresize = from_linear(fres)
    return imgresize


def main():
    args = parse_args()

    indir = args.inpath
    outdir = args.outpath

    sigma0 = 1.6
    o = 4
    s = 3

    # generate base images with sigme=1.6
    folder = os.path.exists(outdir)
    if not folder:
        os.makedirs(outdir)

    firstImgPath = outdir + "/{}".format(sigma0)
    f = os.path.exists(firstImgPath)
    if not f:
        os.makedirs(firstImgPath)

    sigma_orig = 0.5  # the original image is considered as sigma=0.5 blurred
    kernel_size = (0, 0)
    for filename in os.listdir(indir):
        img = cv2.imread(indir + '/' + filename)
        fimg = to_linear(img)
        fres = cv2.resize(fimg, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        fblur = cv2.GaussianBlur(fres, kernel_size, (sigma0 ** 2.0 - (2 * sigma_orig) ** 2.0) ** .5)
        imgblur = from_linear(fblur)
        # show_cv_image(imgblur)
        new_imgName = firstImgPath + '/' + filename.rstrip(".JPG") + ".png"
        cv2.imwrite(new_imgName, imgblur)
    print('1.6 done')

    origin_pics_folder = outdir + "/{}".format(sigma0)
    pics = glob.glob(origin_pics_folder + '/*.png')
    pics.sort()
    # print(origin_pics_folder)
    # print(pics)

    total_sigmas = getSigmas()
    c=0
    for p_path in pics:
        c+=1
        pic_pyramid = list()
        pic_base = cv2.imread(p_path)
        # cv2.imshow('pic_base', pic_base)
        # cv2.waitKey()
        for octave in range(o):
            sigmas = total_sigmas[octave]
            pic_layer = [pic_base]
            for layer in range(s + 2):
                pic_layer.append(getBlur(pic_layer[-1], sigma_now=sigmas[layer], sigma_aim=sigmas[layer + 1]))
                # print(sigmas[layer+1])
            pic_base = getResize(pic_layer[-3], 0.5, 0.5)
            pic_pyramid.append(pic_layer)
        print('{}/{} pyramid done: {}'.format(c, len(pics), p_path))
        for l in range(len(pic_pyramid)):
            for i in range(1, 4):
                # cv2.imshow('pyramid', pic_pyramid[l][i])
                # cv2.waitKey()
                save_path = outdir + '/{}'.format(total_sigmas[l][i]) + '/'
                f = os.path.exists(save_path)
                if not f:
                    os.makedirs(save_path)
                cv2.imwrite(save_path + p_path.split('/')[-1].rstrip('.JPG'), pic_pyramid[l][i])
        print('save done')


if __name__ == "__main__":
    main()
    # Usage:
    # --inpath /cv/projects/boneDescriptors/projects/bones/colmap/Basle/Ailurus_fulgens/dense_2000/images --outpath ../out/af/scale
    # --inpath /cv/projects/boneDescriptors/projects/bones/colmap/Basle/Apteryx_owenii/dense/images --outpath ../out/ao/scale

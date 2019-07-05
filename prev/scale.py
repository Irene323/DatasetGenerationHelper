import numpy as np
import argparse
from PIL import Image, ImageFilter
import os
import cv2
import math
import tempfile
import IPython.display
from prev.srgb import to_linear, from_linear

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


def main():
    args = parse_args()
    separator = "/"

    indir = args.inpath
    outdir = args.outpath
    base = np.array([0., 1. / 3, 2. / 3])
    exp = np.array([2 ** b for b in base])
    print(exp)
    sigma = 1.6 * exp
    print("sigma")
    print(sigma)

    folder = os.path.exists(outdir)
    if not folder:
        os.makedirs(outdir)

    index = np.array([1, 2, 3])
    for t0 in np.arange(5):  # t0 octave
        for ind in index:  # ind layer
            fpath = outdir + separator + "%d_" % t0 + "%d" % ind
            f = os.path.exists(fpath)
            if not f:
                os.makedirs(fpath)

    kernel_size = (0, 0)
    for filename in os.listdir(indir):
        # img = Image.open(indir + separator + filename)
        img = cv2.imread(indir + separator + filename)

        for t10, sigma1 in zip(index, sigma):
            fimg = to_linear(img)
            dst = cv2.GaussianBlur(fimg, kernel_size, sigma1)
            fres = cv2.resize(dst, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            fres = from_linear(fres)
            show_cv_image(fres)
            new_imgName = outdir + separator + '0_%d' % t10 + separator + filename
            cv2.imwrite(new_imgName, fres)

            # # previous way of upsampling
            # shapenew = (dst.shape[1] * 2, dst.shape[0] * 2)
            # dst = cv2.resize(dst, shapenew, interpolation=cv2.INTER_CUBIC)
            # new_imgName = outdir + separator + '0_%d' % t10 + separator + filename
            # cv2.imwrite(new_imgName, dst)

        for t20, sigma2 in zip(index, sigma):
            fimg = to_linear(img)
            dst = cv2.GaussianBlur(fimg, kernel_size, sigma2)
            dst = from_linear(dst)
            show_cv_image(dst)
            new_imgName = outdir + separator + '1_%d' % t20 + separator + filename
            print(new_imgName)
            cv2.imwrite(new_imgName, dst)

        for t30, sigma3 in zip(index, sigma):
            fimg = to_linear(img)
            dst = cv2.GaussianBlur(fimg, kernel_size, sigma3)
            fres = cv2.resize(dst, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            fres = from_linear(fres)
            show_cv_image(fres)
            new_imgName = outdir + separator + '2_%d' % t30 + separator + filename
            cv2.imwrite(new_imgName, fres)

            # shapenew = (int(dst.shape[1] / 2), int(dst.shape[0] / 2))
            # dst = cv2.resize(dst, shapenew, interpolation=cv2.INTER_CUBIC)
            # new_imgName = outdir + separator + '2_%d' % t30 + separator + filename
            # print(new_imgName)
            # cv2.imwrite(new_imgName, dst)

        for t40, sigma4 in zip(index, sigma):
            fimg = to_linear(img)
            dst = cv2.GaussianBlur(fimg, kernel_size, sigma4)
            fres = cv2.resize(dst, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
            fres = from_linear(fres)
            show_cv_image(fres)
            new_imgName = outdir + separator + '3_%d' % t40 + separator + filename
            cv2.imwrite(new_imgName, fres)

            # shapenew = (int(dst.shape[1] / 4), int(dst.shape[0] / 4))
            # dst = cv2.resize(dst, shapenew, interpolation=cv2.INTER_CUBIC)
            # print(new_imgName)
            # new_imgName = outdir + separator + '3_%d' % t40 + separator + filename
            # cv2.imwrite(new_imgName, dst)

        for t50, sigma5 in zip(index, sigma):
            fimg = to_linear(img)
            dst = cv2.GaussianBlur(fimg, kernel_size, sigma5)
            fres = cv2.resize(dst, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_CUBIC)
            fres = from_linear(fres)
            show_cv_image(fres)
            new_imgName = outdir + separator + '4_%d' % 50 + separator + filename
            cv2.imwrite(new_imgName, fres)

            # shapenew = (int(dst.shape[1] / 8), int(dst.shape[0] / 8))
            # dst = cv2.resize(dst, shapenew, interpolation=cv2.INTER_CUBIC)
            # print(new_imgName)
            # new_imgName = outdir + separator + '4_%d' % t50 + separator + filename
            # cv2.imwrite(new_imgName, dst)
    print("done")


if __name__ == "__main__":
    main()
    # Usage:
    # --inpath /cv/projects/boneDescriptors/projects/bones/colmap/Basle/Ailurus_fulgens/dense_2000/images --outpath ../out/scale
import sys
import sqlite3
import numpy as np
import os
import argparse
import math

IS_PYTHON3 = sys.version_info[0] >= 3


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database.db", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    db = sqlite3.connect(args.database_path)
    outdir = args.output_path

    keypoints = dict(
        (image_id, blob_to_array(data, np.float32, (rows, cols)))
        for image_id, rows, cols, data in db.execute("SELECT image_id, rows, cols, data FROM keypoints"))

    # 0, 1,   2,   3,   4,   5
    # x, y, a11, a12, a21, a22
    # ScaleX = sqrt(a11 * a11 + a21 * a21)
    # ScaleY = sqrt(a12 * a12 + a22 * a22)
    # Orientation = atan2(a21, a11)
    # Shear = atan2(-a12, a22) - Orientation
    for i in range(1, len(keypoints) + 1):
        with open(outdir + '/{}.txt'.format(i+2042-1), 'w') as wk:
            for j in range(0, len(keypoints[i])):
                wk.write(str(j+1) + '\t' + \
                         str(keypoints[i][j][0]) + '\t' + \
                         str(keypoints[i][j][1]) + '\t' + \
                         str((keypoints[i][j][2] ** 2 + keypoints[i][j][4] ** 2) ** 0.5) + '\t' + \
                         str((keypoints[i][j][3] ** 2 + keypoints[i][j][5] ** 2) ** 0.5) + '\t' + \
                         str(math.atan2(keypoints[i][j][4], keypoints[i][j][2])) + '\t' + \
                         str(math.atan2(-keypoints[i][j][3], keypoints[i][j][5]) - \
                             math.atan2(keypoints[i][j][4], keypoints[i][j][2])) + '\n')

    # descriptors = dict(
    #     (image_id, blob_to_array(data, np.uint8, (rows, cols)))
    #     for image_id, rows, cols, data in db.execute("SELECT image_id, rows, cols, data FROM descriptors"))
    # # print(descriptors[1].shape)
    #
    # for i in range(1, len(descriptors) + 1):
    #     np.savetxt(outdir + '/{}.txt'.format(i+2042-1), descriptors[i])

if __name__ == "__main__":
    main()

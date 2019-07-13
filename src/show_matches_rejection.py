import sys
import sqlite3
import numpy as np
import os
import argparse
import math
import cv2

IS_PYTHON3 = sys.version_info[0] >= 3


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database.db", required=True)
    parser.add_argument("--first_id", required=True)
    args = parser.parse_args()

    db = sqlite3.connect(args.database_path)
    first_id = int(args.first_id)

    keypoints = dict(
        (image_id, blob_to_array(data, np.float32, (rows, cols)))
        for image_id, rows, cols, data in db.execute("SELECT image_id, rows, cols, data FROM keypoints"))

    img = cv2.imread("/cv/projects/boneDescriptors/projects/bones/colmap/Basle/Ailurus_fulgens/dense_2000/images/IMG_2042.JPG")
    with open("/home/yirenli/dev/DatasetGeneration/out/af/filter_test/colmap.txt", "r") as r:
        for l in r:
            a=l.strip("\n").split("\t")
            print(a)
            imangeID0 = int(a[0].split("_")[0])
            featureID0 = int(a[0].split("_")[1])
            print(featureID0)
            kpInfo = [keypoints[1][featureID0][0],keypoints[1][featureID0][1]]
            print(kpInfo)
            if a[2]=="0":
                cv2.circle(img, (kpInfo[0], kpInfo[1]), 3, (60,20,220))
            elif a[2]=="1":
                cv2.circle(img, (kpInfo[0], kpInfo[1]), 3, (250,230,230))
            elif a[2]=="2":
                cv2.circle(img, (kpInfo[0], kpInfo[1]), 3, (255,255,0))
            elif a[2]=="3":
                cv2.circle(img, (kpInfo[0], kpInfo[1]), 3, (0,215,255))
            elif a[2]=="4":
                cv2.circle(img, (kpInfo[0], kpInfo[1]), 3, (211,0,148))
            elif a[2]=="-1":
                cv2.circle(img, (kpInfo[0], kpInfo[1]), 3, (0,255,0))
    cv2.imwrite("/home/yirenli/dev/DatasetGeneration/out/af/filter_test/test.png", img)

if __name__ == "__main__":
    main()
    # Usage:
    # --database_path /home/yirenli/data/databases/af_2000.db --first_id 2042

    # 0 - no feature found in the radius: too large scale
    # 1 - no feature found in the radius: too close to the edge
    # 2 - multiple features in the radius
    # 3 - didn't pass position or depth threshold
    # 4 - didn't pass rotation or scale check
    # -1 - match
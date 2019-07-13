import sys
import sqlite3
import numpy as np
import argparse
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

    for i in range(1, len(keypoints) + 1):
        img = cv2.imread(
            "/cv/projects/boneDescriptors/projects/bones/colmap/Basle/Ailurus_fulgens/dense_2000/images/IMG_{}.JPG".format(
                i + first_id - 1))
        for j in range(0, len(keypoints[i])):
            cv2.circle(img, (keypoints[i][j][0], keypoints[i][j][1]), 0, (0, 0, 255))
        if (i == 6):
            cv2.imwrite("/home/yirenli/{}.png".format(6 + first_id - 1), img)
            break
        # cv2.imshow(str(i+first_id-1), img)
        # cv2.waitKey()


if __name__ == "__main__":
    main()
    # Usage:
    # --database_path /home/yirenli/data/databases/ao_2000.db --keypoints_path ../out/ao/keypoints --descriptors_path ../out/ao/descriptors --first_id 2309

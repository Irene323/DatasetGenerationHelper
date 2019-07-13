import sys
import sqlite3
import numpy as np
import os
import argparse
import math

MAX_IMAGE_ID = 2 ** 31 - 1

IS_PYTHON3 = sys.version_info[0] >= 3

def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database.db", required=True)
    parser.add_argument("--matchlist_path", required=True)
    args = parser.parse_args()

    matchlist_path = args.matchlist_path
    db = sqlite3.connect(args.database_path)

    matches = dict(
        (pair_id_to_image_ids(pair_id),
         blob_to_array(data, np.uint32, (-1, 2)))
        for pair_id, data in db.execute("SELECT pair_id, data FROM matches")
    )

    # print(matches[(1,2)])

    with open(matchlist_path, "w") as w:
        for pair in matches[(1,2)]:
            w.write("2042\t{}\t2043\t{}\n".format(pair[0]+1, pair[1]+1))


if __name__ == "__main__":
    main()

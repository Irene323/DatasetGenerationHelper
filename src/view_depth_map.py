# COLMAP - Structure-from-Motion and Multi-View Stereo.
# Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# COLMAP - Structure-from-Motion and Multi-View Stereo.
# Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pylab as plt


def read_array(path):
    print(path)
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


def main():
    # depth_map = read_array("/cv/projects/boneDescriptors/projects/bones/colmap/Basle/Ailurus_fulgens/dense_2000/stereo/depth_maps/IMG_2049.JPG.geometric.bin")
    depth_map = read_array("/cv/projects/boneDescriptors/projects/bones/colmap/Basle/Apteryx_owenii/dense/stereo/depth_maps/IMG_2309.JPG.geometric.bin")

    # Visualize the depth map.
    min_depth, max_depth = np.percentile(depth_map, [0, 95])

    print("min depth: ", min_depth, " max depth:", max_depth)
    min_depth = 0
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth

    plt.imshow(depth_map)
    plt.show()


if __name__ == "__main__":
    main()
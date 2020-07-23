import os
import sys
import h5py
import numpy as np
import argparse
import open3d as o3d
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--root', help='path to root directory')
args = parser.parse_args()

root = args.root
file_list = os.listdir(os.path.join(root, 'raw_data'))
my_data = open(os.path.join(root, 'metadata', 'my_data.txt'), 'w')
my_test = open(os.path.join(root, 'metadata', 'my_test.txt'), 'w')

for file_name in file_list:
    fname = os.path.join(root, 'raw_data', file_name)
    pcd = o3d.io.read_point_cloud(fname)

    pdb.set_trace()
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    added = np.concatenate((points, colors), axis=1)
    fname = os.path.join(root, 'npy', file_name.strip('.ply'))
    np.save(fname, added)
    my_data.write(file_name.strip('.ply') + '.npy' + '\n')
    my_test.write(file_name.strip('.ply') + '.h5' + '\n')





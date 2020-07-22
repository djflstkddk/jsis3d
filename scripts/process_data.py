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
name = 'teddy_bear.ply'
fname = os.path.join(root, 'raw_data', name)
pcd = o3d.io.read_point_cloud(fname)

pdb.set_trace()
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
added = np.concatenate((points, colors), axis = 1)
fname = os.path.join(root, 'npz', name.strip('.ply'))
np.save(fname, added)






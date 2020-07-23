import numpy as np
import argparse
import open3d as o3d
import pdb
import h5py
import os
import copy
import matplotlib as mpl

parser = argparse.ArgumentParser()
parser.add_argument('--root', default = '/home/dvision/PycharmProjects/jsis3d/', help='path to root directory')
args = parser.parse_args()
batch_size = 8
num_points = 4096
root = args.root
file_list = os.listdir(os.path.join(root, 'data', 's3dis', 'my_h5'))

pdict = np.load(os.path.join(root, 'logs', 'my_s3dis', 'pred.npz'))
pdict = np.stack([pdict['semantics'], pdict['instances']], axis=-1)
pdict = np.concatenate(pdict, axis=0)
offset = 0
classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
color_cate = [[153,0,0], [204,102,0], [153,153,0], [76,153,0], [0,204,0], [0,153,76], [0,204,204],
              [0,76,153], [0,0,204], [76,0,153], [204,0,204], [153,0,76], [64,64,64]]
color_cate = [[x/255 for x in color] for color in color_cate]

for file_name in file_list:
    data = h5py.File(os.path.join(root, 'data', 's3dis', 'my_h5', file_name))
    _pdict = pdict[offset:offset+batch_size*num_points]
    print("file_name: {}".format(file_name))
    for i in range(13):
        print("class : {}, number : {}".format(classes[i], sum(_pdict[:,0]==i)))

    points = data['coords']
    colors = data['points'][:, :, 3:6]
    points = np.concatenate(points, axis=0)
    colors = np.concatenate(colors, axis=0)
    seg_colors = copy.deepcopy(colors)
    for i in range(13):
        seg_colors[_pdict[:,0]==i]=color_cate[i]

    #pdb.set_trace()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points) # before elimination
    pcd.colors = o3d.utility.Vector3dVector(seg_colors) # segmentation visualization
    o3d.visualization.draw_geometries([pcd])

    pcd.colors = o3d.utility.Vector3dVector(colors) # original colors
    o3d.visualization.draw_geometries([pcd])
    idx = np.logical_and(_pdict[:, 0] != 1, _pdict[:, 0] != 10) # omit floor and bookcase points
    print("bef :{} , after:{}".format( idx.shape[0], sum(idx)))
    points = points[idx]
    colors = colors[idx]
    pcd.points = o3d.utility.Vector3dVector(points) # after elimination
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
    offset += batch_size*num_points

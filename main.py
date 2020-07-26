import numpy as np
import argparse
import open3d as o3d
import pdb
import h5py
import os
import copy
import matplotlib as mpl

parser = argparse.ArgumentParser()
parser.add_argument('--root', default = './', help='path to root directory')
args = parser.parse_args()
#batch_size = 72 # batch_size depends on how many windows a scene(.ply file) has when making .h5 files
num_points = 4096
root = args.root
file_list = os.path.join(root, 'data', 's3dis', 'metadata', 'my_test.txt')
flist = [line.strip() for line in open(file_list)]

pdict = np.load(os.path.join(root, 'logs', 's3dis', 'my_pred.npz'))
pdict = np.stack([pdict['semantics'], pdict['instances']], axis=-1)

offset = 0
classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
color_cate = [[153,0,0], [204,102,0], [153,153,0], [76,153,0], [0,204,0], [0,153,76], [0,204,204],
              [0,76,153], [0,0,204], [76,0,153], [204,0,204], [153,0,76], [64,64,64]]
color_cate = [[x/255 for x in color] for color in color_cate]

for file_name in flist:
    print("file name : {}".format(file_name))
    data = h5py.File(os.path.join(root, 'data', 's3dis', 'my_h5', file_name))
    points = data['coords'][:]
    colors = data['points'][:, :, 3:6]
    #labels = data['labels'][:]
    batch_size = points.shape[0]
    _pdict = pdict[offset:offset+batch_size]
    #pdb.set_trace()
    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    #labels = labels.reshape(-1, 2)
    _pdict = _pdict.reshape(-1, 2)
    pdb.set_trace()

    print("file_name : {}, batch_size : {}".format(file_name, batch_size))
    for i in range(13):
        print("class : {}, number : {}".format(classes[i], sum(_pdict[:,0]==i)))
    print()
    #for i in range(13):
    #    print("class : {}, number : {}".format(classes[i], sum(labels[:,0]==i)))
    """
    for i in range(13):
        indices = (truth[:, 0] == i)
        correct = (pred[indices, 0] == truth[indices, 0])
        accu[i]  += np.sum(correct)
        freq[i]  += np.sum(indices)
        inter[i] += np.sum((pred[:, 0] == i) & (truth[:, 0] == i))
        union[i] += np.sum((pred[:, 0] == i) | (truth[:, 0] == i))
    """
    pdb.set_trace()


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
    offset += batch_size

import os
import sys
import h5py
import numpy as np
import argparse
import pdb


parser = argparse.ArgumentParser()
parser.add_argument('--root', default='../data/s3dis', help='path to root directory')
parser.add_argument('--seed', type=int, default=42, help='random number seed')
args = parser.parse_args()

root = args.root
seed = args.seed
np.random.seed(seed)
fname = os.path.join(root, 'metadata', 'classes.txt')
classes = [line.strip() for line in open(fname, 'r')]

fname = os.path.join(root, 'metadata', 'my_data.txt')
flist = [os.path.join(root, 'npy', line.strip())
         for line in open(fname, 'r')]


def sample_cloud(cloud, num_samples):
    n = cloud.shape[0]
    if n >= num_samples:
        indices = np.random.choice(n, num_samples, replace=False)
    else:
        indices = np.random.choice(n, num_samples - n, replace=True)
        indices = list(range(n)) + list(indices)
    sampled = cloud[indices, :]
    return sampled


def room_to_blocks(fname, num_points, size=1.0, stride=0.5, threshold=100):
    cloud = np.load(fname)
    #cloud[:, 3:6] /= 255.0
    #pdb.set_trace()
    trans = np.array([[0,0,1], [-1,0,0], [0,-1,0]])
    coords = np.transpose(cloud[:, :3])
    coords = np.transpose(np.matmul(trans, coords))

    limit_min = np.asarray([-3.8, -3.14, -1.2])
    coords = coords - limit_min
    cloud[:, 0:3] = coords

    limit = np.asarray([5.55, 7.75, 2.5])  # limit should be the room size. If the point cloud is part of a room, type limit manually.

    width = int(np.ceil((limit[0] - size) / stride)) + 1
    depth = int(np.ceil((limit[1] - size) / stride)) + 1
    cells = [(x * stride, y * stride) for x in range(width) for y in range(depth)]
    blocks = []
    for (x, y) in cells:
        xcond = (cloud[:, 0] <= x + size) & (cloud[:, 0] >= x)
        ycond = (cloud[:, 1] <= y + size) & (cloud[:, 1] >= y)
        cond  = xcond & ycond
        if np.sum(cond) < threshold:
            continue
        block = cloud[cond, :]
        block = sample_cloud(block, num_points)
        blocks.append(block)
    blocks = np.stack(blocks, axis=0)
    # A batch should have shape of BxNx14, where
    # [0:3] - global coordinates
    # [3:6] - block normalized coordinates (centered at Z-axis)
    # [6:9] - RGB colors
    # [9:12] - room normalized coordinates
    # [12:14] - semantic and instance labels
    num_blocks = blocks.shape[0]
    batch = np.zeros((num_blocks, num_points, 14))
    for b in range(num_blocks):
        minx = min(blocks[b, :, 0])
        miny = min(blocks[b, :, 1])
        batch[b, :, 3]  = blocks[b, :, 0] - (minx + size * 0.5)
        batch[b, :, 4]  = blocks[b, :, 1] - (miny + size * 0.5)
        batch[b, :, 9]  = blocks[b, :, 0] / limit[0]
        batch[b, :, 10] = blocks[b, :, 1] / limit[1]
        batch[b, :, 11] = blocks[b, :, 2] / limit[2]
    batch[:,:, 0:3] = blocks[:,:,0:3]
    batch[:,:, 5:9] = blocks[:,:,2:6]
    #batch[:,:, 12:] = blocks[:,:,6:8]  # when processing cumstomized data, there are no labels.
    return batch


def save_batch_h5(fname, batch):
    fp = h5py.File(fname)
    coords = batch[:, :, 0:3]
    points = batch[:, :, 3:12]
    #labels = batch[:, :, 12:14]
    fp.create_dataset('coords', data=coords, compression='gzip', dtype='float32')
    fp.create_dataset('points', data=points, compression='gzip', dtype='float32')
    #fp.create_dataset('labels', data=labels, compression='gzip', dtype='int64')
    fp.close()


for fname in flist:
    basename = os.path.basename(fname).strip('.npy')
    if not os.path.exists(fname):
        print('[WARNING] Cannot find {}'.format(fname))
        continue

    #pdb.set_trace()
    num_points = 4096
    batch = room_to_blocks(fname, num_points, size=1.0, stride=0.5)
    fname = os.path.join(root, 'my_h5', basename + '.h5')
    print('> Saving batch to {}...'.format(fname))
    if not os.path.exists(fname):
        save_batch_h5(fname, batch)

import tensorflow as tf
import scipy.io as sio
import numpy as np

import matplotlib.pyplot as plt

def visualize_voxel(voxel):
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxel, facecolors='red', edgecolor='k')

    plt.show()

def num_vertices(filename):

    file1 = open(filename, 'r')
    lines = file1.readlines()
    num_vertices = 0

    for line in lines:
        if line[0] == 'v':
            num_vertices += 1

    return num_vertices

def num_faces(filename):

    file1 = open(filename, 'r')
    lines = file1.readlines()
    num_faces = 0

    for line in lines:
        if line[0] == 'f':
            num_faces += 1

    return num_faces

def downsample(vox_in, n, use_max=True):
    """
    Downsample a 3-d tensor n times
    Inputs:
      - vox_in (Tensor): HxWxD tensor
      - n (int): number of times to downsample each dimension
      - use_max (bool): use maximum value when downsampling. If set to False
                        the mean value is used.
    Output:
      - vox_out (Tensor): (H/n)x(W/n)x(D/n) tensor
    """
    dimy = vox_in.shape[0] // n
    dimx = vox_in.shape[1] // n
    dimz = vox_in.shape[2] // n
    vox_out = np.zeros((dimy, dimx, dimz))
    for x in range(dimx):
        for y in range(dimy):
            for z in range(dimz):
                subx = x * n
                suby = y * n
                subz = z * n
                subvox = vox_in[suby : suby + n, subx : subx + n, subz : subz + n]
                if use_max:
                    vox_out[y, x, z] = np.amax(subvox)
                else:
                    vox_out[y, x, z] = np.mean(subvox)
    return vox_out

def verts2voxel(verts, voxel_size):
    def valid_coords(x, y, z, vx_size):
        Hv, Wv, Zv = vx_size
        indx = (x >= 0) * (x < Wv)
        indy = (y >= 0) * (y < Hv)
        indz = (z >= 0) * (z < Zv)
        return indx * indy * indz

    Hv, Wv, Zv = voxel_size
    # create original voxel of size VxVxV
    # orig_voxel = tf.cast(np.zeros((Hv, Wv, Zv)), tf.float32)
    orig_voxel = (np.zeros((Hv, Wv, Zv)))

    x = (verts[:, 0] + 1) * (Wv - 1) / 2
    x = x.astype(np.int32)
    y = (verts[:, 1] + 1) * (Hv - 1) / 2
    y = y.astype(np.int32)
    z = (verts[:, 2] + 1) * (Zv - 1) / 2
    z = z.astype(np.int32)

    keep = valid_coords(x, y, z, voxel_size)
    x = x[keep]
    y = y[keep]
    z = z[keep]

    orig_voxel[(y), (x), (z)] = 1.0

    # align with image coordinate system
    flip_idx = (list(range(Hv)[::-1]))
    orig_voxel = np.take(orig_voxel, indices=flip_idx, axis=0)

    flip_idx = (list(range(Wv)[::-1]))
    orig_voxel = np.take(orig_voxel, indices=flip_idx, axis=1)
    return tf.cast(orig_voxel, tf.float32)


def read_voxel(voxelfile):
    """
    Reads voxel and transforms it in the form of verts
    """
    #Path manager used in original implementation
    with open(voxelfile, "rb") as f:
        voxel = sio.loadmat(f)["voxel"]
    voxel = np.rot90(voxel, k=3, axes=(1, 2))
    verts = np.argwhere(voxel > 0).astype(np.float32, copy=False)

    # centering and normalization
    min_x = np.min(verts[:, 0])
    max_x = np.max(verts[:, 0])
    min_y = np.min(verts[:, 1])
    max_y = np.max(verts[:, 1])
    min_z = np.min(verts[:, 2])
    max_z = np.max(verts[:, 2])
    verts[:, 0] = verts[:, 0] - (max_x + min_x) / 2
    verts[:, 1] = verts[:, 1] - (max_y + min_y) / 2
    verts[:, 2] = verts[:, 2] - (max_z + min_z) / 2
    scale = np.sqrt(np.max(np.sum(verts ** 2, axis=1))) * 2
    verts /= scale
    verts = tf.cast(verts, dtype=tf.float32)

    return verts


def transform_verts(verts, R, t):
    """
    Transforms verts with rotation R and translation t
    Inputs:
        - verts (tensor): of shape (N, 3)
        - R (tensor): of shape (3, 3) or None
        - t (tensor): of shape (3,) or None
    Outputs:
        - rotated_verts (tensor): of shape (N, 3)
    """
    rot_verts = verts.clone().t()
    if R is not None:
        assert R.dim() == 2
        assert R.size(0) == 3 and R.size(1) == 3
        rot_verts = torch.mm(R, rot_verts)
    if t is not None:
        assert t.dim() == 1
        assert t.size(0) == 3
        rot_verts = rot_verts + t.unsqueeze(1)
    rot_verts = rot_verts.t()
    return rot_verts

def apply_coords(coords, new_w, new_h):
    coords = np.array(coords)
    coords[:, 0] = coords[:, 0] * (new_w * 1.0 / coords.shape[0])
    coords[:, 1] = coords[:, 1] * (new_h * 1.0 / coords.shape[1])
    return coords

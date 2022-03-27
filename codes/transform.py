#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Jinyuan Sun
# @Time    : 2022/3/27 8:59 AM
# @File    : transform.py
# @annotation    :

import numpy as np
from scipy.spatial.transform import Rotation as R
from Bio import PDB


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def transform(amino_acid):
    """Rotate amino acid to local coordinate CA is (0,0,0), CA_C is y-axis
    param amino_acid: A 3d array of amino acid coordinates
    return amino_acid: Rotated amino acid
    """
    # transform CA to (0,0,0)
    amino_acid -= amino_acid[1]
    unit_y = np.array([0, 1, 0])
    atom_C = amino_acid[2]
    r1 = rotation_matrix_from_vectors(atom_C, unit_y)
    # print(f'rotation matrix:{r1}')
    r1 = R.from_matrix(r1)
    # rotation CA_C to y-axis
    amino_acid = np.array(r1.apply(amino_acid))

    atom_N_xz_proj = np.array([amino_acid[0][0], 0, amino_acid[0][2]])
    unit_x = np.array([1, 0, 0])

    # r_ang = test_ser[0].dot(target_xy)/(np.linalg.norm(test_ser[0]) * np.linalg.norm(target_xy))
    r_ang = unit_x.dot(atom_N_xz_proj) / (np.linalg.norm(unit_x) * np.linalg.norm(atom_N_xz_proj))
    # print(f'rotation angle:{r_ang}')

    if amino_acid[0][2] < 0:
        r2 = R.from_euler('y', -np.degrees(np.arccos(r_ang)), degrees=True)
    else:
        r2 = R.from_euler('y', np.degrees(np.arccos(r_ang)), degrees=True)
    amino_acid = np.array(r2.apply(amino_acid))

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    return amino_acid


def rigid_transform_3d(A, B):
    """ Find the rotation  and translation matrix of point set A and B
    :param A: A 3d "source" vector
    :param B: A 3d "destination" vector
    :return R: A rotation matrix (3x3)
    :return t: A translation vector
    Align A to B: np.matmul(A, R.T) + t.reshape([1, 3])
    """
    assert len(A) == len(B), "Invalid length"  # Length of two vector should be the same

    N = A.shape[0]  # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    H = np.matmul(np.transpose(AA), BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = np.matmul(Vt.T, U.T)

    t = -np.matmul(R, centroid_A) + centroid_B
    # err = B - np.matmul(A,R.T) - t.reshape([1, 3])
    return R, t


def gly_cb(residue):
    """
    param: residue: is a biopython object
    return: cd: cd coordinate
    """
    # get atom coordinates as vectors
    n = residue["N"].get_vector()
    c = residue["C"].get_vector()
    ca = residue["CA"].get_vector()
    # center at origin
    n = n - ca
    c = c - ca
    # find rotation matrix that rotates n
    # -120 degrees along the ca-c vector
    rot = PDB.vectors.rotaxis(-np.pi * 120.0 / 180.0, c)
    # apply rotation to ca-n vector
    cb_at_origin = n.left_multiply(rot)
    # put on top of ca atom
    cb = cb_at_origin + ca
    return cb


if __name__ == '__main__':
    # print(f'ser1: {transform(test_ser1)}')
    # print(f'ser2: {transform(test_ser2)}')
    # transform(test_ser2)

    # Input: expects Nx3 matrix of points
    # Returns R,t
    # R = 3x3 rotation matrix
    # t = 3x1 column vector

    test_ser1 = np.array([[-23.051, 13.053, 29.892],
                          [-22.939, 11.648, 29.542],
                          [-24.200, 11.154, 28.862],
                          [-25.300, 11.620, 29.169],
                          [-22.592, 10.823, 30.772],
                          [-21.225, 11.005, 31.093]])

    test_ser2 = np.array([[-17.835, 21.999, 29.736],
                          [-18.233, 20.926, 30.662],
                          [-18.291, 19.538, 30.012],
                          [-17.945, 19.361, 28.821],
                          [-19.604, 21.256, 31.222],
                          [-19.680, 20.857, 32.572]])

    R_mat, t = rigid_transform_3d(test_ser1[:3], test_ser2[:3])

    print(np.matmul(test_ser1, R_mat.T) + t.reshape([1, 3]))

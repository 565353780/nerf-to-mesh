#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse


def getFLAGS():
    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-di', '--display-interval', type=int, default=0)
    parser.add_argument('-mr', '--min-roughness', type=float, default=0.08)
    parser.add_argument('-mip',
                        '--custom-mip',
                        action='store_true',
                        default=False)
    parser.add_argument('-bg',
                        '--background',
                        default='checker',
                        choices=['black', 'white', 'checker', 'reference'])
    parser.add_argument('--loss',
                        default='logl1',
                        choices=['logl1', 'logl2', 'mse', 'smape', 'relmse'])
    parser.add_argument('-o', '--out-dir', type=str, default=None)
    parser.add_argument('--validate', type=bool, default=True)

    FLAGS = parser.parse_args()

    FLAGS.mtl_override = None  # Override material of model
    FLAGS.dmtet_grid = 64  # Resolution of initial tet grid. We provide 64 and 128 resolution grids. Other resolutions can be generated with https://github.com/crawforddoran/quartet
    FLAGS.mesh_scale = 2.1  # Scale of tet grid box. Adjust to cover the model
    FLAGS.env_scale = 1.0  # Env map intensity multiplier
    FLAGS.envmap = None  # HDR environment probe
    FLAGS.display = None  # Conf validation window/display. E.g. [{"relight" : <path to envlight>}]
    FLAGS.camera_space_light = False  # Fixed light in camera space. This is needed for setups like ethiopian head where the scanned object rotates on a stand.
    FLAGS.lock_light = False  # Disable light optimization in the second pass
    FLAGS.lock_pos = False  # Disable vertex position optimization in the second pass
    FLAGS.sdf_regularizer = 0.2  # Weight for sdf regularizer (see paper for details)
    FLAGS.laplace = "relative"  # Mesh Laplacian ["absolute", "relative"]
    FLAGS.laplace_scale = 10000.0  # Weight for sdf regularizer. Default is relative with large weight
    FLAGS.pre_load = True  # Pre-load entire dataset into memory for faster training
    FLAGS.kd_min = [0.0, 0.0, 0.0, 0.0]  # Limits for kd
    FLAGS.kd_max = [1.0, 1.0, 1.0, 1.0]
    FLAGS.ks_min = [0.0, 0.08, 0.0]  # Limits for ks
    FLAGS.ks_max = [1.0, 1.0, 1.0]
    FLAGS.nrm_min = [-1.0, -1.0, 0.0]  # Limits for normal map
    FLAGS.nrm_max = [1.0, 1.0, 1.0]
    FLAGS.cam_near_far = [0.1, 1000.0]
    FLAGS.learn_light = True
    FLAGS.multi_gpu = False
    FLAGS.local_rank = 0
    return FLAGS

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../nvdiffrec")

import os
import json
import torch
import argparse
import nvdiffrast.torch as dr
from render import light, obj
from geometry.dlmesh import DLMesh
from geometry.dmtet import DMTetGeometry
from train import initial_guess_material, optimize_mesh, xatlas_uvmap

from nerf_to_mesh.Dataset.nerf import DatasetNERF

RADIUS = 3.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r',
                        '--train-res',
                        nargs=2,
                        type=int,
                        default=[512, 512])
    parser.add_argument('-tr',
                        '--texture-res',
                        nargs=2,
                        type=int,
                        default=[1024, 1024])
    parser.add_argument('-di', '--display-interval', type=int, default=0)
    parser.add_argument('-si', '--save-interval', type=int, default=1000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)
    parser.add_argument('-mr', '--min-roughness', type=float, default=0.08)
    parser.add_argument('-mip',
                        '--custom-mip',
                        action='store_true',
                        default=False)
    parser.add_argument('-rt',
                        '--random-textures',
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
    parser.add_argument('-rm', '--ref_mesh', type=str)
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

    data = json.load(open(FLAGS.config, 'r'))
    for key in data:
        FLAGS.__dict__[key] = data[key]

    FLAGS.display_res = FLAGS.train_res

    os.makedirs(FLAGS.out_dir, exist_ok=True)

    glctx = dr.RasterizeGLContext()

    assert os.path.isdir(FLAGS.ref_mesh)
    assert os.path.isfile(os.path.join(FLAGS.ref_mesh,
                                       'transforms_train.json'))
    assert os.path.isfile(os.path.join(FLAGS.ref_mesh, 'transforms_test.json'))
    dataset_train = DatasetNERF(os.path.join(FLAGS.ref_mesh,
                                             'transforms_train.json'),
                                FLAGS,
                                examples=(FLAGS.iter + 1) * FLAGS.batch)
    dataset_validate = DatasetNERF(
        os.path.join(FLAGS.ref_mesh, 'transforms_test.json'), FLAGS)

    lgt = light.create_trainable_env_rnd(512, scale=0.0, bias=0.5)

    geometry = DMTetGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS)

    mat = initial_guess_material(geometry, True, FLAGS)

    geometry, mat = optimize_mesh(glctx,
                                  geometry,
                                  mat,
                                  lgt,
                                  dataset_train,
                                  dataset_validate,
                                  FLAGS,
                                  pass_idx=0,
                                  pass_name="dmtet_pass1",
                                  optimize_light=FLAGS.learn_light)

    base_mesh = xatlas_uvmap(glctx, geometry, mat, FLAGS)

    torch.cuda.empty_cache()
    mat['kd_ks_normal'].cleanup()
    del mat['kd_ks_normal']

    lgt = lgt.clone()
    geometry = DLMesh(base_mesh, FLAGS)

    os.makedirs(os.path.join(FLAGS.out_dir, "dmtet_mesh"), exist_ok=True)
    obj.write_obj(os.path.join(FLAGS.out_dir, "dmtet_mesh/"), base_mesh)
    light.save_env_map(os.path.join(FLAGS.out_dir, "dmtet_mesh/probe.hdr"),
                       lgt)

    geometry, mat = optimize_mesh(glctx,
                                  geometry,
                                  base_mesh.material,
                                  lgt,
                                  dataset_train,
                                  dataset_validate,
                                  FLAGS,
                                  pass_idx=1,
                                  pass_name="mesh_pass",
                                  warmup_iter=100,
                                  optimize_light=FLAGS.learn_light
                                  and not FLAGS.lock_light,
                                  optimize_geometry=not FLAGS.lock_pos)

    final_mesh = geometry.getMesh(mat)
    os.makedirs(os.path.join(FLAGS.out_dir, "mesh"), exist_ok=True)
    obj.write_obj(os.path.join(FLAGS.out_dir, "mesh/"), final_mesh)
    light.save_env_map(os.path.join(FLAGS.out_dir, "mesh/probe.hdr"), lgt)

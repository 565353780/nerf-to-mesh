#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../nvdiffrec")

import os
import torch
import nvdiffrast.torch as dr
from render import light, obj
from geometry.dlmesh import DLMesh
from geometry.dmtet import DMTetGeometry
from train import initial_guess_material, optimize_mesh, xatlas_uvmap

from nerf_to_mesh.Config.config import getFLAGS

from nerf_to_mesh.Dataset.nerf import DatasetNERF

if __name__ == "__main__":
    FLAGS = getFLAGS()

    FLAGS.ref_mesh = "/home/chli/chLi/NeRF/ustc_niu"
    FLAGS.train_res = [1280, 720]

    FLAGS.random_textures = True
    FLAGS.iter = 5000
    FLAGS.save_interval = 100
    FLAGS.texture_res = [2048, 2048]
    FLAGS.batch = 1
    FLAGS.learning_rate = [0.03, 0.01]
    FLAGS.ks_min = [0, 0.25, 0]
    FLAGS.dmtet_grid = 128
    FLAGS.mesh_scale = 2.3
    FLAGS.laplace_scale = 3000
    FLAGS.display = [{"latlong" : True}, {"bsdf" : "kd"}, {"bsdf" : "ks"}, {"bsdf" : "normal"}]
    FLAGS.layers = 8
    FLAGS.background = "white"
    FLAGS.out_dir = "./out/ustc_niu/"

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

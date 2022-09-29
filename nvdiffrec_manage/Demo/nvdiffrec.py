#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../nvdiffrec")

import os
import time
import torch
import numpy as np
import nvdiffrast.torch as dr
from render import light, obj, util
from geometry.dmtet import DMTetGeometry
from geometry.dlmesh import DLMesh
from train import createLoss, prepare_batch, validate_itr, initial_guess_material, xatlas_uvmap

from nvdiffrec_manage.Config.config import getFLAGS

from nvdiffrec_manage.Dataset.nerf import DatasetNERF

from nvdiffrec_manage.Module.trainer import Trainer


def optimize_mesh(glctx,
                  geometry,
                  opt_material,
                  lgt,
                  dataset_train,
                  dataset_validate,
                  FLAGS,
                  warmup_iter=0,
                  log_interval=10,
                  pass_idx=0,
                  pass_name="",
                  optimize_light=True,
                  optimize_geometry=True):

    learning_rate = FLAGS.learning_rate[pass_idx] if isinstance(
        FLAGS.learning_rate, list) or isinstance(
            FLAGS.learning_rate, tuple) else FLAGS.learning_rate
    learning_rate_pos = learning_rate[0] if isinstance(
        learning_rate, list) or isinstance(learning_rate,
                                           tuple) else learning_rate
    learning_rate_mat = learning_rate[1] if isinstance(
        learning_rate, list) or isinstance(learning_rate,
                                           tuple) else learning_rate

    def lr_schedule(iter, fraction):
        if iter < warmup_iter:
            return iter / warmup_iter
        return max(0.0, 10**(
            -(iter - warmup_iter) *
            0.0002))  # Exponential falloff from [1.0, 0.1] over 5k epochs.

    image_loss_fn = createLoss(FLAGS)

    trainer_noddp = Trainer(glctx, geometry, lgt, opt_material,
                            optimize_geometry, optimize_light, image_loss_fn,
                            FLAGS)

    trainer = trainer_noddp
    if optimize_geometry:
        optimizer_mesh = torch.optim.Adam(trainer_noddp.geo_params,
                                          lr=learning_rate_pos)
        scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(
            optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9))

    optimizer = torch.optim.Adam(trainer_noddp.params, lr=learning_rate_mat)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9))

    img_cnt = 0
    img_loss_vec = []
    reg_loss_vec = []
    iter_dur_vec = []

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=FLAGS.batch,
        collate_fn=dataset_train.collate,
        shuffle=True)
    dataloader_validate = torch.utils.data.DataLoader(
        dataset_validate, batch_size=1, collate_fn=dataset_train.collate)

    def cycle(iterable):
        iterator = iter(iterable)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(iterable)

    v_it = cycle(dataloader_validate)

    for it, target in enumerate(dataloader_train):

        target = prepare_batch(target, 'random')

        display_image = FLAGS.display_interval and (it % FLAGS.display_interval
                                                    == 0)
        save_image = FLAGS.save_interval and (it % FLAGS.save_interval == 0)
        if display_image or save_image:
            result_image, result_dict = validate_itr(
                glctx, prepare_batch(next(v_it), FLAGS.background), geometry,
                opt_material, lgt, FLAGS)
            np_result_image = result_image.detach().cpu().numpy()
            if display_image:
                util.display_image(np_result_image,
                                   title='%d / %d' % (it, FLAGS.iter))
            if save_image:
                util.save_image(
                    FLAGS.out_dir + '/' + ('img_%s_%06d.png' %
                                           (pass_name, img_cnt)),
                    np_result_image)
                img_cnt = img_cnt + 1

        iter_start_time = time.time()

        optimizer.zero_grad()
        if optimize_geometry:
            optimizer_mesh.zero_grad()

        img_loss, reg_loss = trainer(target, it)

        total_loss = img_loss + reg_loss

        img_loss_vec.append(img_loss.item())
        reg_loss_vec.append(reg_loss.item())

        total_loss.backward()
        if hasattr(lgt,
                   'base') and lgt.base.grad is not None and optimize_light:
            lgt.base.grad *= 64
        if 'kd_ks_normal' in opt_material:
            opt_material['kd_ks_normal'].encoder.params.grad /= 8.0

        optimizer.step()
        scheduler.step()

        if optimize_geometry:
            optimizer_mesh.step()
            scheduler_mesh.step()

        with torch.no_grad():
            if 'kd' in opt_material:
                opt_material['kd'].clamp_()
            if 'ks' in opt_material:
                opt_material['ks'].clamp_()
            if 'normal' in opt_material:
                opt_material['normal'].clamp_()
                opt_material['normal'].normalize_()
            if lgt is not None:
                lgt.clamp_(min=0.0)

        torch.cuda.current_stream().synchronize()
        iter_dur_vec.append(time.time() - iter_start_time)

        if it % log_interval == 0:
            img_loss_avg = np.mean(np.asarray(img_loss_vec[-log_interval:]))
            reg_loss_avg = np.mean(np.asarray(reg_loss_vec[-log_interval:]))
            iter_dur_avg = np.mean(np.asarray(iter_dur_vec[-log_interval:]))

            remaining_time = (FLAGS.iter - it) * iter_dur_avg
            print(
                "iter=%5d, img_loss=%.6f, reg_loss=%.6f, lr=%.5f, time=%.1f ms, rem=%s"
                % (it, img_loss_avg, reg_loss_avg,
                   optimizer.param_groups[0]['lr'], iter_dur_avg * 1000,
                   util.time_to_text(remaining_time)))

    return geometry, opt_material


def demo():
    FLAGS = getFLAGS()

    FLAGS.ref_mesh = "/home/chli/chLi/NeRF/ustc_niu_merge_10"
    FLAGS.train_res = [1280, 720]
    FLAGS.iter = 5000
    FLAGS.save_interval = 100
    FLAGS.learning_rate = [0.03, 0.01]
    FLAGS.out_dir = "./out/ustc_niu_merge_10/"

    FLAGS.display_res = FLAGS.train_res
    os.makedirs(FLAGS.out_dir, exist_ok=True)

    glctx = dr.RasterizeGLContext()

    assert os.path.isdir(FLAGS.ref_mesh)
    assert os.path.isfile(os.path.join(FLAGS.ref_mesh, 'transform.json'))
    dataset_train = DatasetNERF(os.path.join(FLAGS.ref_mesh, 'transform.json'),
                                FLAGS,
                                examples=(FLAGS.iter + 1) * FLAGS.batch)
    dataset_validate = DatasetNERF(
        os.path.join(FLAGS.ref_mesh, 'transform.json'), FLAGS)

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
    return True

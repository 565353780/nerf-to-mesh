#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../nerfmeshes/src")

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.memory import ModelSummary

import models
from lightning_modules import LoggerCallback, PathParser

from nerf_to_mesh.Method.nerfmeshes.colmap_convert import gen_poses


def demo():
    scenedir = "/home/chli/chLi/NeRF/ustc_niu_nerfmeshes/"

    gen_poses(scenedir)

    torch.set_printoptions(threshold=100,
                           edgeitems=50,
                           precision=8,
                           sci_mode=False)

    config = "./Config/nerfmeshes/ustc_niu.yml"
    log_checkpoint = "./output/ustc_niu/"
    checkpoint = None
    run_name = "ustc_niu"
    gpus = 1
    precision = 32

    # Log path
    path_parser = PathParser()
    cfg, logger = path_parser.parse(config,
                                    log_checkpoint,
                                    run_name,
                                    checkpoint,
                                    create_logger=True)

    model = getattr(models, cfg.experiment.model)(cfg)

    checkpoint_callback = ModelCheckpoint(filepath=path_parser.checkpoint_dir,
                                          save_top_k=3,
                                          save_last=True,
                                          verbose=True,
                                          monitor="val_loss",
                                          mode="min",
                                          prefix="model_")

    logger_callback = LoggerCallback(cfg)

    trainer = Trainer(weights_summary=None,
                      resume_from_checkpoint=path_parser.checkpoint_path,
                      gpus=gpus,
                      default_root_dir=path_parser.log_dir,
                      logger=logger,
                      num_sanity_val_steps=0,
                      checkpoint_callback=checkpoint_callback,
                      row_log_interval=1,
                      log_gpu_memory=None,
                      precision=precision,
                      profiler=None,
                      fast_dev_run=False,
                      deterministic=False,
                      progress_bar_refresh_rate=0,
                      accumulate_grad_batches=1,
                      callbacks=[logger_callback])

    logger.experiment.add_text("description", cfg.experiment.description, 0)
    logger.experiment.add_text("config",
                               f"\t{cfg.dump()}".replace("\n", "\n\t"), 0)
    logger.experiment.add_text(
        "params",
        f"\t{ModelSummary(model, mode='full')}".replace("\n", "\n\t"), 0)

    trainer.fit(model)

    print("Done!")

    return True

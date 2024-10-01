'''
Author: Jiaxin Zheng
Date: 2023-08-31 10:29:37
LastEditors: Jiaxin Zheng
LastEditTime: 2024-10-01 19:47:05
Description: 
'''
import json
import ast
import os
from typing import Any, Dict, List, Tuple
import pandas as pd

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    # assert cfg.ckpt_path

    # log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    # log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    output_dir=os.path.join(trainer.log_dir,'test_results')
    
    for file in cfg.data.test.non_dynamic.file:
        data_cfg=cfg.data
        data_cfg.test.non_dynamic.file=[file]
        task=file.split('/')[-1].split('.')[0]
        
        log.info(f"Instantiating datamodule <{cfg.data._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

        log.info(f"Starting testing! {file}")

        model.test_pred={}
        
        trainer.test(model=model, datamodule=datamodule,ckpt_path=cfg.ckpt_path)

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        # df
        pred_df = trainer.model.pred_df
        pred_df.to_csv(os.path.join(output_dir,f'{task}.csv'),index=False)
        
        if trainer.model.scores is not None:
            scores = trainer.model.scores
            save_file_path = os.path.join(output_dir,f'{task}.json')
            with open(save_file_path,'w') as f:
                json.dump(scores,f)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }


    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_main.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()

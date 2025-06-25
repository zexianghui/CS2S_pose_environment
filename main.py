import os,sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MASTER_PORT'] = '5812'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch    
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from utils.util import instantiate_from_config
from collections import OrderedDict

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--base", nargs="*", metavar="base_config.yaml",help="paths to base configs. Loaded from left-to-right. Parameters can be overwritten or added with command-line options of the form `--key value`.", default=list())
    # parser.add_argument("--devices", type=str, default="0,", nargs="?")
    # parser.add_argument("--strategy", type=str, default="ddp", nargs="?")
    parser.add_argument("--seed", type=int, default=24, help="seed for seed_everything")
    parser.add_argument("--scale_lr", type=str2bool, nargs="?", const=True, default=True, help="scale base-lr by ngpu * batch_size * n_accumulate")
    parser.add_argument("--logdir", type=str, const=True, default="result", nargs="?", help="logdir")

    parser.add_argument("--train", type=str2bool, const=True, default=False, nargs="?", help="train")
    parser.add_argument("--resume", type=str, const=True, default="", nargs="?", help="resume from logdir or checkpoint in logdir")
    
    parser.add_argument("--test", type=str, default="", nargs="?")
    return parser

if __name__ == "__main__":
    #get parser
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()
    seed_everything(opt.seed)
    cfg_fname = os.path.split(opt.base[0])[-1]
    cfg_name = os.path.splitext(cfg_fname)[0]
    name = "_" + cfg_name
    nowname = now + name
    logdir = os.path.join(opt.logdir, nowname)

    #resume
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        
        paths = opt.resume.split("/")
        logdir = "/".join(paths[:-2])
        ckpt = opt.resume
        opt.resume_from_checkpoint = ckpt
    
    if opt.test:
        paths = opt.test.split("/")
        logdir = "/".join(paths[:-2]) + '/test'

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())

    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config["accelerator"] = "gpu"
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    if not "devices" in trainer_config:
        del trainer_config["accelerator"]
        cpu = True
    else:
        cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config


    #log
    trainer_kwargs = dict()
    default_callbacks_cfg = {
            "setup_callback": {
                "target": "utils.callback.callback.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                    "testckpt_path": opt.test
                }
            },
            "image_logger": {
                "target": "utils.callback.callback.ImageLogger",
                "params": {
                    "batch_frequency": 3000,
                    "max_images": 4,
                    "logdir": os.path.join(logdir, 'image'),
                    "clamp": True
                }
            },
            # "learning_rate_logger": {
            #     "target": "main.LearningRateMonitor",
            #     "params": {
            #         "logging_interval": "step",
            #         # "log_momentum": True
            #     }
            # },
        }
    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    trainer_kwargs["logger"] = False
    trainer_kwargs["enable_checkpointing"] = False
    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir

    try:
        # dataset
        data = instantiate_from_config(config.data)
        # data.prepare_data()
        # data.setup()
        # print("#### Data #####")
        # for k in data.datasets:
        #     print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # model
        model = instantiate_from_config(config.model)
        #learning ratel
        base_lr = config.model.base_learning_rate
        model.learning_rate = base_lr


        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")


        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()


        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # checkpoint = torch.load("result/2024-01-19T01-31-06_Boost_Sat2Den_diffusion_condition/checkpoints/epoch_45.ckpt", map_location=torch.device('cpu'))
        # model.load_state_dict(checkpoint['state_dict'])
        # run
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        ngpu = len(lightning_config.trainer.devices.strip(",").split(','))
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        model.learning_rate = 7.0e-05
        print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))

        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if opt.test and not trainer.interrupted:
            trainer.test(model, data, ckpt_path = opt.test)
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        print(trainer.profiler.summary())

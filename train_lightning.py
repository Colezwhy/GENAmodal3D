import os

n_thread = 2
os.environ["MKL_NUM_THREADS"] = f"{n_thread}" 
os.environ["NUMEXPR_NUM_THREADS"] = f"{n_thread}" 
os.environ["OMP_NUM_THREADS"] = f"4" 
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{n_thread}" 
os.environ["OPENBLAS_NUM_THREADS"] = f"{n_thread}" 


import torch
from dataLoader import dataset_dict
from omegaconf import OmegaConf

# from lightning.system_vae import system
from lightning.system import system
from torch.utils.data import DataLoader
import pytorch_lightning as L

from datetime import datetime


from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from argparse import ArgumentParser
import sys
import random
import torch.distributed as dist

class UpdateViewsCallback(L.Callback):
    def __init__(self, n_views_list=[1, 2, 3]):
        self.n_views_list = n_views_list

    def on_train_epoch_start(self, trainer, pl_module):
        # 当前 epoch 编号
        epoch = trainer.current_epoch

        # 只在 rank 0 决定 view 数量
        if trainer.global_rank == 0:
            # 根据 epoch 顺序选取视角数量（循环）
            idx = epoch % len(self.n_views_list)
            new_n = self.n_views_list[idx]

            # 保存为 tensor
            new_n_tensor = torch.tensor([new_n], device=pl_module.device)
        else:
            # 其他 rank 初始化一个占位 tensor
            new_n_tensor = torch.tensor([0], device=pl_module.device)

        # 所有 GPU 同步 new_n
        import torch.distributed as dist
        dist.broadcast(new_n_tensor, src=0)

        # 获取广播后的值
        new_n = new_n_tensor.item()

        # 更新 dataset
        dataset = trainer.train_dataloader.dataset
        dataset.update_views(new_n)

        if trainer.global_rank == 0:
            print(f"[INFO]: Epoch {epoch} → updated training sparse views to {new_n}")


def check_weights_dtype(weights):
    for key, value in weights.items():
        print(f"{key}: {value.dtype}")

def main(cfg, ckpt, cfg_path):

    torch.set_float32_matmul_precision('medium')
    torch.autograd.set_detect_anomaly(True)
    print("Using PyTorch {} and Lightning {}".format(torch.__version__, L.__version__))

    # data loader
    train_dataset = dataset_dict[cfg.train_dataset.dataset_name]
    train_loader = DataLoader(train_dataset(cfg.train_dataset), 
                              batch_size= cfg.train.batch_size,
                              num_workers= 8, 
                              shuffle=True,
                              pin_memory=False)
    val_dataset = dataset_dict[cfg.test_dataset.dataset_name]
    val_loader = DataLoader(val_dataset(cfg.test_dataset), 
                              batch_size=cfg.test.batch_size,
                              num_workers=2,
                              shuffle=True,
                              pin_memory=False)
    
    # build logger
    project_name = cfg.exp_name.split("/")[0]
    exp_name = cfg.exp_name.split("/")[1]

    if cfg.logger.name == "tensorboard":
        logger = TensorBoardLogger(save_dir=cfg.logger.dir, name=exp_name)
    elif cfg.logger.name == "wandb":
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        logger = WandbLogger(name=exp_name,project=project_name, save_dir=cfg.logger.dir, entity="large-reconstruction-model")
    
    # Set up ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.vis_dir,        # Path where checkpoints will be saved
        filename='{epoch}',        # Filename for the checkpoints
        # save_top_k=1,             # Set to -1 to save all checkpoints
        every_n_epochs=1,          # Save a checkpoint every K epochs
        save_on_train_epoch_end=True,  # Ensure it saves at the end of an epoch, not the beginning
    )
    # DYNAMIC TRAINING BATCH
    update_views_callback = UpdateViewsCallback(n_views_list=[1, 2, 3])

    print(cfg.vis_dir)
    my_system = system(cfg, cfg_path, full_train=cfg.full_train)
    max_epoch = cfg.train.n_epoch
        

    trainer = L.Trainer(devices=cfg.gpu_id,
                        num_nodes=1,
                        max_epochs=max_epoch,
                        accelerator='gpu',
                        strategy=DDPStrategy(find_unused_parameters=False),
                        accumulate_grad_batches=8,
                        logger=logger,
                        precision="16-mixed",
                        callbacks=[checkpoint_callback, update_views_callback],
                        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
                        limit_val_batches=cfg.train.limit_val_batches,  # Run on only 10% of the validation data
                        limit_train_batches=cfg.train.limit_train_batches, 
                        )
    

    t0 = datetime.now()
    trainer.fit(
        my_system, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt
        )
    
    dt = datetime.now() - t0
    print('Training took {}'.format(dt))


if __name__ == '__main__':

    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, default = "./configs/base.yaml")
    parser.add_argument("--ckpt", type=str, default = None)
    parser.add_argument("--vis_dir", type=str, default = None)
    parser.add_argument("--fulltrain", default = False, action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    
    base_conf = OmegaConf.load(args.config)

    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(base_conf, cli_conf)

    if args.vis_dir is not None:
        cfg.vis_dir = args.vis_dir

    if args.strict:
        cfg.strict = True
    else:
        cfg.strict = False
        
    if args.fulltrain:
        cfg.full_train = True
    else:
        cfg.full_train = False
    
    main(cfg, args.ckpt, args.config)
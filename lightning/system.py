
import torch
import numpy as np
import pytorch_lightning as L

import torch.nn as nn
from lightning.utils import vis_images, CosineWarmupScheduler

from lightning.network_ss import Network_ss
from lightning.network_slat import Network_slat

import matplotlib.pyplot as plt
import os
import shutil

class system(L.LightningModule):
    def __init__(self, cfg, cfg_path=None, full_train=False):
        super().__init__()

        self.cfg = cfg
        if cfg.model.type == 'ss':
            self.net = Network_ss(cfg, full_train=full_train)
        elif cfg.model.type == 'slat':
            self.net = Network_slat(cfg, full_train=full_train)

        self.validation_step_outputs = []

        self.vis_dir = cfg.vis_dir
        os.makedirs(self.vis_dir, exist_ok=True)
        if cfg_path is not None:
            shutil.copy(cfg_path, os.path.join(self.vis_dir, 'config.yaml'))

    def training_step(self, batch, batch_idx):
        with torch.amp.autocast('cuda'):
            output, loss = self.net(batch)
        if output is not None:
            if 0 == self.trainer.global_step % 1000  and (self.trainer.local_rank == 0):
                self.vis_results(output, batch, prex='train')
        self.log(f'rec', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.net.eval()
        output, loss = self.net(batch, if_vis=True)
        if output is not None:
            if batch_idx == 0 and (self.trainer.local_rank == 0):
                self.vis_results(output, batch, prex='val')
        self.validation_step_outputs.append({'rec': loss})
        return loss

    def on_validation_epoch_end(self):
        keys = self.validation_step_outputs[0]
        for key in keys:
            prog_bar = True if key in ['psnr','mask','depth', 'rec'] else False
            metric_mean = torch.stack([x[key] for x in self.validation_step_outputs]).mean()
            self.log(f'val/{key}', metric_mean, prog_bar=prog_bar, sync_dist=True)

        self.validation_step_outputs.clear()  # free memory
        torch.cuda.empty_cache()

    def vis_results(self, output, batch, prex):
        output_vis = vis_images(output, batch)
        for key, value in output_vis.items():
            imgs = [np.concatenate([img for img in value],axis=0)]
            for idx, img in enumerate(imgs):
                img_save_path = os.path.join(self.vis_dir, f'{prex}_{self.global_step}_{key}.png')
                plt.imsave(img_save_path, img)
        self.net.train()
        if hasattr(self.net, 'img_encoder'):
            self.net.img_encoder.eval()

    def num_steps(self) -> int:
        """Get number of steps"""
        # Accessing _data_source is flaky and might break
        dataset = self.trainer.fit_loop._data_source.dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, self.trainer.num_devices)
        num_steps = dataset_size * self.trainer.max_epochs * self.cfg.train.limit_train_batches // (self.trainer.accumulate_grad_batches * num_devices)
        return int(num_steps)

    def configure_optimizers(self):
        decay_params, no_decay_params = [], []

        # add all bias and LayerNorm params to no_decay_params
        for name, module in self.named_modules():
            if isinstance(module, nn.LayerNorm):
                no_decay_params.extend([p for p in module.parameters()])
            elif hasattr(module, 'bias') and module.bias is not None:
                no_decay_params.append(module.bias)

        # add remaining parameters to decay_params
        _no_decay_ids = set(map(id, no_decay_params))
        decay_params = [p for p in self.parameters() if id(p) not in _no_decay_ids]

        # filter out parameters with no grad
        decay_params = list(filter(lambda p: p.requires_grad, decay_params))
        no_decay_params = list(filter(lambda p: p.requires_grad, no_decay_params))

        # Optimizer
        opt_groups = [
            {'params': decay_params, 'weight_decay': self.cfg.train.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(
            opt_groups,
            lr=self.cfg.train.lr,
            betas=(self.cfg.train.beta1, self.cfg.train.beta2),
        )

        total_global_batches = self.num_steps()
        scheduler = CosineWarmupScheduler(
                        optimizer=optimizer,
                        warmup_iters=self.cfg.train.warmup_iters,
                        max_iters=total_global_batches,
                    )

        return {"optimizer": optimizer, 
                "lr_scheduler": {
                'scheduler': scheduler,
                'interval': 'step'  # or 'epoch' for epoch-level updates
            }}
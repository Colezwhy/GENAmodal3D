import torch.nn as nn
import torch
import numpy as np
from gena3d.modules.sparse import SparseTensor

def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == 'warm0.1':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.2':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.5':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    else:
        raise NotImplementedError(schedule_type)
    return betas


class GaussianDiffusion:
    def __init__(self,betas, loss_type, model_mean_type, model_var_type):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

        self.sigma_min = 1e-5

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (
                self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )

    def q_sample_linear(self, x_start, t, noise=None):
        """
        x_t = (1 - t) * x0 + t * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        t_shape = [x_start.shape[0]] + [1] * (len(x_start.shape) - 1)
        t = t.view(*t_shape)
        return (1 - t) * x_start + t * noise

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )

    

    '''losses'''

    def p_losses(self, denoise_fn, data_start, t, noise=None, y=None, multi_t=False):
        """
        Training loss calculation
        """
        # print(data_start.shape)
        if multi_t:
            B = len(data_start['feats'])
            assert t.shape == torch.Size([B])

            noise, data_feats_t = [], []
            for idx, feat in enumerate(data_start['feats']):
                noise_single = torch.randn(feat.shape, dtype=feat.dtype, device=feat.device)
                noise.append(noise_single)
                assert noise_single.shape == feat.shape and noise_single.dtype == feat.dtype
                data_feat_t = self.q_sample_linear(x_start=feat.unsqueeze(0), t=t[idx:idx+1], noise=noise_single.unsqueeze(0))
                data_feats_t.append(data_feat_t.squeeze(0))
            data_feats_t = torch.cat(data_feats_t, dim=0)
            data_coords_t = torch.cat(data_start['coords'], dim=0)

            data_t = SparseTensor(
                feats=data_feats_t.half().contiguous(), 
                coords=data_coords_t.contiguous()
            )

            # predict the noise instead of x_start. seems to be weighted naturally like SNR
            eps_recon = denoise_fn(data_t, t, y).feats
            target = torch.cat(noise, dim=0) - torch.cat(data_start['feats'], dim=0)
            losses = ((eps_recon - target)**2).mean()

            # assert losses.shape == torch.Size([B])
            return losses, {'data_t': data_t, 'eps_recon': eps_recon}
        else:
            B = 1
            assert t.shape == torch.Size([B])

            if noise is None:
                for idx, feat in enumerate(data_start['feats']):
                    noise_single = torch.randn(feat.shape, dtype=feat.dtype, device=feat.device)
                    noises.append(noise_single)
                noise = torch.cat(noises, dim=0)

            data_start_sp = SparseTensor(
                feats=torch.cat(data_start['feats'], dim=0).contiguous(),
                coords=torch.cat(data_start['coords'], dim=0).contiguous()
            )

            assert noise.shape == data_start_sp.feats.shape and noise.dtype == data_start_sp.feats.dtype

            data_t_feats = self.q_sample_linear(x_start=data_start_sp.feats.unsqueeze(0), t=t, noise=noise.unsqueeze(0))
            data_t = SparseTensor(
                feats=data_t_feats.squeeze(0).contiguous(),
                coords=data_start_sp.coords.contiguous()
            )

            eps_recon = denoise_fn(data_t, t, y).feats
            target = noise - torch.cat(data_start['feats'], dim=0)
            losses = ((eps_recon - target)**2).mean()

            # assert losses.shape == torch.Size([B])
            return losses, {'data_t': data_t, 'eps_recon': eps_recon}
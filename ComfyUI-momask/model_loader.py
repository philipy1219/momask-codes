import os
import sys
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)
from os.path import join as pjoin
import torch

def load_vq_model(opt, device):
    from models.vq.model import RVQVAE
    
    vq_model = RVQVAE(opt,
                      opt.dim_pose,
                      opt.nb_code,
                      opt.code_dim,
                      opt.output_emb_width,
                      opt.down_t,
                      opt.stride_t,
                      opt.width,
                      opt.depth,
                      opt.dilation_growth_rate,
                      opt.vq_act,
                      opt.vq_norm)
    
    ckpt = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', 'net_best_fid.tar'),
                      map_location=device)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {opt.name} Completed!')
    return vq_model

def load_transformer_model(opt, device, which_model='latest.tar'):
    from models.mask_transformer.transformer import MaskTransformer
    
    t2m_transformer = MaskTransformer(code_dim=opt.code_dim,
                                    cond_mode='text',
                                    latent_dim=opt.latent_dim,
                                    ff_size=opt.ff_size,
                                    num_layers=opt.n_layers,
                                    num_heads=opt.n_heads,
                                    dropout=opt.dropout,
                                    clip_dim=512,
                                    cond_drop_prob=opt.cond_drop_prob,
                                    clip_version='ViT-B/32',
                                    opt=opt)
    
    ckpt = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', which_model),
                      map_location=device)
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

def load_residual_model(opt, vq_opt, device):
    from models.mask_transformer.transformer import ResidualTransformer
    
    opt.num_quantizers = vq_opt.num_quantizers
    opt.num_tokens = vq_opt.nb_code
    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
                                        cond_mode='text',
                                        latent_dim=opt.latent_dim,
                                        ff_size=opt.ff_size,
                                        num_layers=opt.n_layers,
                                        num_heads=opt.n_heads,
                                        dropout=opt.dropout,
                                        clip_dim=512,
                                        shared_codebook=vq_opt.shared_codebook,
                                        cond_drop_prob=opt.cond_drop_prob,
                                        share_weight=opt.share_weight,
                                        clip_version='ViT-B/32',
                                        opt=opt)

    ckpt = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', 'net_best_fid.tar'),
                      map_location=device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer

def load_length_estimator(opt, device):
    from models.vq.model import LengthEstimator
    
    model = LengthEstimator(512, 50)
    ckpt = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'length_estimator', 'model', 'finest.tar'),
                      map_location=device)
    model.load_state_dict(ckpt['estimator'])
    print(f'Loading Length Estimator from epoch {ckpt["epoch"]}!')
    return model 
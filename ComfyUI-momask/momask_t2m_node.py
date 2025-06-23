import os
from os.path import join as pjoin
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical
import sys
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from model_loader import (
    load_vq_model,
    load_transformer_model,
    load_residual_model,
    load_length_estimator
)
from .utils import recover_from_ric
from visualization.joints2bvh import Joint2BVHConvertor

class MoMaskText2MotionNode:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_version = 'ViT-B/32'
        self.output_dir = "outputs/momask_t2m"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化配置
        self.init_config()
        
        # 加载模型
        self.load_models()
        
    def init_config(self):
        # VQ模型配置
        self.vq_config = {
            "checkpoints_dir": "checkpoints",
            "dataset_name": "humanml",
            "name": "vq_model",
            "dim_pose": 263,  # humanml数据集使用263维
            "nb_code": 512,
            "code_dim": 512,
            "output_emb_width": 512,
            "down_t": 2,
            "stride_t": 2,
            "width": 512,
            "depth": 3,
            "dilation_growth_rate": 3,
            "vq_act": "relu",
            "vq_norm": "none"
        }
        
        # Transformer模型配置
        self.transformer_config = {
            "checkpoints_dir": "checkpoints",
            "dataset_name": "humanml",
            "name": "transformer_model",
            "code_dim": 512,
            "latent_dim": 512,
            "ff_size": 1024,
            "n_layers": 8,
            "n_heads": 8,
            "dropout": 0.1,
            "cond_drop_prob": 0.25
        }
        
        # Residual模型配置
        self.residual_config = {
            "checkpoints_dir": "checkpoints",
            "dataset_name": "humanml",
            "name": "residual_model",
            "latent_dim": 512,
            "ff_size": 1024,
            "n_layers": 8,
            "n_heads": 8,
            "dropout": 0.1,
            "cond_drop_prob": 0.25,
            "shared_codebook": True,
            "share_weight": True
        }
        
    def load_models(self):
        # 创建配置对象
        class Config:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)
        
        vq_opt = Config(self.vq_config)
        transformer_opt = Config(self.transformer_config)
        residual_opt = Config(self.residual_config)
        
        # 加载VQ模型
        self.vq_model = load_vq_model(vq_opt, self.device)
        
        # 加载Transformer模型
        self.t2m_transformer = load_transformer_model(transformer_opt, self.device)
        
        # 加载Residual模型
        self.res_model = load_residual_model(residual_opt, vq_opt, self.device)
        
        # 加载Length Estimator
        self.length_estimator = load_length_estimator(transformer_opt, self.device)
        
        # 将所有模型移到设备上并设置为评估模式
        for model in [self.vq_model, self.t2m_transformer, self.res_model, self.length_estimator]:
            model.to(self.device)
            model.eval()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_prompt": ("STRING", {
                    "multiline": True,
                    "default": "A person is walking"
                }),
                "motion_length": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "step": 1
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "cond_scale": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1
                }),
                "time_steps": ("INT", {
                    "default": 18,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "topk_filter_thres": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "gumbel_sample": ("BOOLEAN", {
                    "default": False
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("bvh_path",)
    FUNCTION = "generate_motion"
    CATEGORY = "momask"

    def generate_motion(self, text_prompt, motion_length=0, temperature=1.0, cond_scale=5.0, time_steps=18, topk_filter_thres=0.9, gumbel_sample=False):
        with torch.no_grad():
            # 估计动作长度
            if motion_length == 0:
                text_embedding = self.t2m_transformer.encode_text([text_prompt])
                pred_dis = self.length_estimator(text_embedding)
                probs = F.softmax(pred_dis, dim=-1)
                token_lens = Categorical(probs).sample()
            else:
                token_lens = torch.LongTensor([motion_length // 4]).to(self.device).long()

            m_length = token_lens * 4

            # 生成动作
            mids = self.t2m_transformer.generate([text_prompt], token_lens,
                                               timesteps=time_steps,
                                               cond_scale=cond_scale,
                                               temperature=temperature,
                                               topk_filter_thres=topk_filter_thres,
                                               gsample=gumbel_sample)
            
            mids = self.res_model.generate(mids, [text_prompt], token_lens,
                                         temperature=temperature,
                                         cond_scale=cond_scale)
            
            pred_motions = self.vq_model.forward_decoder(mids)
            pred_motions = pred_motions.detach().cpu().numpy()
            
            # 反归一化
            mean = np.load(pjoin(self.vq_config["checkpoints_dir"], 
                               self.vq_config["dataset_name"], 
                               self.vq_config["name"], 
                               "meta/mean.npy"))
            std = np.load(pjoin(self.vq_config["checkpoints_dir"], 
                              self.vq_config["dataset_name"], 
                              self.vq_config["name"], 
                              "meta/std.npy"))
            data = pred_motions * std + mean

            # 处理动作数据
            joint_data = data[0][:m_length[0]]
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()

            # 转换为BVH
            converter = Joint2BVHConvertor()
            bvh_path = os.path.join(self.output_dir, f"motion_{hash(text_prompt)}.bvh")
            _, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)

            return (bvh_path,)

NODE_CLASS_MAPPINGS = {
    "MoMaskText2Motion": MoMaskText2MotionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MoMaskText2Motion": "Text to Motion (MoMask)"
} 
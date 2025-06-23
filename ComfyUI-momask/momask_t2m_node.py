import os
from os.path import join as pjoin
from argparse import Namespace
import re
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical
import sys
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(
    base_path
)

from model_loader import (
    load_vq_model,
    load_transformer_model,
    load_residual_model,
    load_length_estimator
)
from .utils import recover_from_ric
from visualization.joints2bvh import Joint2BVHConvertor

POS_enumerator = {
    'VERB': 0,
    'NOUN': 1,
    'DET': 2,
    'ADP': 3,
    'NUM': 4,
    'AUX': 5,
    'PRON': 6,
    'ADJ': 7,
    'ADV': 8,
    'Loc_VIP': 9,
    'Body_VIP': 10,
    'Obj_VIP': 11,
    'Act_VIP': 12,
    'Desc_VIP': 13,
    'OTHER': 14,
}

def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    if str(numStr).isdigit():
        flag = True
    return flag


def get_opt(opt_path, device, **kwargs):
    opt = Namespace()
    opt_dict = vars(opt)

    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    with open(opt_path, 'r') as f:
        for line in f:
            if line.strip() not in skip:
                # print(line.strip())
                key, value = line.strip('\n').split(': ')
                if value in ('True', 'False'):
                    opt_dict[key] = (value == 'True')
                #     print(key, value)
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)

    # print(opt)
    opt_dict['which_epoch'] = 'finest'
    opt.save_root = pjoin(base_path, opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(base_path, opt.save_root, 'model')
    opt.meta_dir = pjoin(base_path, opt.save_root, 'meta')

    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D/'
        opt.motion_dir = pjoin(base_path, opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(base_path, opt.data_root, 'texts')
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 196
        opt.max_motion_frame = 196
        opt.max_motion_token = 55
    elif opt.dataset_name == 'kit':
        opt.data_root = './dataset/KIT-ML/'
        opt.motion_dir = pjoin(base_path, opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(base_path, opt.data_root, 'texts')
        opt.joints_num = 21
        opt.dim_pose = 251
        opt.max_motion_length = 196
        opt.max_motion_frame = 196
        opt.max_motion_token = 55
    else:
        raise KeyError('Dataset not recognized')
    if not hasattr(opt, 'unit_length'):
        opt.unit_length = 4
    opt.dim_word = 300
    opt.num_classes = 200 // opt.unit_length
    opt.dim_pos_ohot = len(POS_enumerator)
    opt.is_train = False
    opt.is_continue = False
    opt.device = device

    opt_dict.update(kwargs) # Overwrite with kwargs params

    return opt

class MoMaskText2MotionNode:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_version = 'ViT-B/32'
        self.output_dir = "outputs/momask_t2m"
        os.makedirs(self.output_dir, exist_ok=True)
        # 加载模型
        self.load_models()
        
    def load_models(self):
        dim_pose = 253
        dataset_name = "t2m"
        checkpoints_dir = "checkpoints"
        name = "t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns"
        root_dir = pjoin(base_path, checkpoints_dir, dataset_name, name)
        model_opt_path = pjoin(base_path, root_dir, 'opt.txt')
        model_opt = get_opt(model_opt_path, device=self.device)
        
        vq_opt_path = pjoin(base_path, checkpoints_dir, dataset_name, model_opt.vq_name, 'opt.txt')
        vq_opt = get_opt(vq_opt_path, device=self.device)
        vq_opt.dim_pose = dim_pose
        vq_opt.checkpoints_dir = pjoin(base_path, vq_opt.checkpoints_dir)

        model_opt.num_tokens = vq_opt.nb_code
        model_opt.num_quantizers = vq_opt.num_quantizers
        model_opt.code_dim = vq_opt.code_dim
        model_opt.checkpoints_dir = pjoin(base_path, model_opt.checkpoints_dir)

        res_opt_path = pjoin(base_path, checkpoints_dir, dataset_name, "tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw", 'opt.txt')
        res_opt = get_opt(res_opt_path, device=self.device)
        res_opt.checkpoints_dir = pjoin(base_path, res_opt.checkpoints_dir)

        assert res_opt.vq_name == model_opt.vq_name
        
        # 加载VQ模型
        self.vq_model = load_vq_model(vq_opt, self.device)
        
        # 加载Transformer模型
        self.t2m_transformer = load_transformer_model(model_opt, self.device)
        
        # 加载Residual模型
        self.res_model = load_residual_model(res_opt, vq_opt, self.device)
        
        # 加载Length Estimator
        self.length_estimator = load_length_estimator(model_opt, self.device)
        
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
            mean = np.load(pjoin(base_path, self.vq_config["checkpoints_dir"], 
                               self.vq_config["dataset_name"], 
                               self.vq_config["name"], 
                               "meta/mean.npy"))
            std = np.load(pjoin(base_path, self.vq_config["checkpoints_dir"], 
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
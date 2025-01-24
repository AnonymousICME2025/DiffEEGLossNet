from diffusion.distillation import *
from diffusion.diffusion import *
from diffusion.eegwave import *
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import json
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'	
def load_checkpoint(savepath, device):				# 定义一个函数，用于加载保存的模型状态
    checkpoint = torch.load(savepath)			    # 加载保存的模型状态
    epoch = checkpoint['epoch']					    # 获取保存的训练轮次
    config = checkpoint['config']				    # 获取保存的模型配置
    function_approximator = EEGWave(			    # 创建EEGWave模型
        checkpoint['n_class'],
        checkpoint['n_subject'],
        checkpoint['N'],
        checkpoint['n'],
        checkpoint['C'],
        checkpoint['E'],
        checkpoint['K']
    )
    model = Diffusion(function_approximator, checkpoint['T'])	# 创建Diffusion模型
    model.load_state_dict(checkpoint['model_state_dict'])		# 加载模型的参数
    model.to(device)						# 将模型移动到指定的设备（GPU或CPU）
    return epoch, config, model				# 返回训练轮次、模型配置和模型


# 定义一个函数，用于生成样本
def sample():
    with open("diffusion/sampling_conf.json",'r') as fconf:		# 打开配置文件
        config = json.load(fconf)			# 读取配置文件的内容
    cp_path = Path(f"{os.path.dirname(os.path.abspath(__file__))}/diffusion/checkpoints/{config['checkpoint']}")		 # 获取保存的模型状态的路径
    epoch, config2, model = load_checkpoint(cp_path, device)			 # 加载保存的模型状态
    flag_class_conditioning = "_c" if config2["CLASS_CONDITIONING"] else ""		# 根据配置文件中的CLASS_CONDITIONING字段，设置对应的标志
    flag_subject_conditioning = "_s" if config2["SUBJECT_CONDITIONING"] else ""	# 根据配置文件中的SUBJECT_CONDITIONING字段，设置对应的标志
    with open (Path(f"{os.path.dirname(os.path.abspath(__file__))}/data/{config['data']}_stats.json"),'r') as fstat:		# 打开统计文件 
        stats = json.load(fstat)												# 读取统计文件的内容
    nb_orig_samples = 4837 if config['data']=="vepess" else 2592				# 根据配置文件中的data字段，设置对应的原始样本数量
    
    # 设置样本路径
    sample_path = Path(f"{os.path.dirname(os.path.abspath(__file__))}/sampled/{config['data']}/{config['checkpoint'][:-4]}{flag_class_conditioning}{flag_subject_conditioning}_{config['set']}")
    sample_path.mkdir(parents=True, exist_ok=True)		    # 创建样本路径
    index_start = max(0,len(os.listdir(sample_path))-1)		 # 获取样本路径中的文件数量
    
    for s in stats:							# 遍历统计文件中的每个主题
        print(f"Subject {s}/{len(stats)}")
        for c in stats[s]:					# 遍历每个主题中的每个类别
            nb_samples_of_class_subject = int(stats[s][c] * config['nb_samples'] / nb_orig_samples)		# 计算每个主题和类别的样本数量
            class_condition = torch.tensor([int(c)],dtype=torch.long,device=device) if flag_class_conditioning else None	# 根据配置文件中的CLASS_CONDITIONING字段，设置对应的类别条件
            subject_condition = torch.tensor([int(s)],dtype=torch.long,device=device) if flag_subject_conditioning else None	# 根据配置文件中的SUBJECT_CONDITIONING字段，设置对应的主题条件
            for index in tqdm(range(index_start, index_start+nb_samples_of_class_subject)):	# 遍历每个样本
                x_hat = model(config['signal_length'], config['gamma'],			# 使用模型生成样本
                    class_conditioning=class_condition, subject_conditioning=subject_condition)
                torch.save((x_hat.detach().cpu(),class_condition,subject_condition), f"{sample_path}/tensor{index}.pt")		# 保存生成的样本
            index_start = index_start + nb_samples_of_class_subject		 # 更新样本的起始索引
    
    config["nb_samples"] = len(os.listdir(sample_path))		# 更新配置文件中的样本数量
    with open(f"{sample_path}/sampling_conf.json",'w') as f:		# 打开配置文件
        json.dump({**config,**config2}, f)		# 将配置文件的内容写入文件

    return sample_path			 # 返回样本路径

# 如果这个脚本被直接运行，而不是被导入，那么就调用sample函数
if __name__ == '__main__':
    sample()
# --coding:utf-8--
import os

from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer
from tqdm import tqdm

import time

import logging
import json

device1=torch.device('cuda:0')
# device2=torch.device('cpu')

input_path = "/orcd/scratch/orcd/009/lxz/WavTokenizer/test_filelist.txt"
out_folder = './result/infer'
# os.system("rm -r %s"%(out_folder))
# os.system("mkdir -p %s"%(out_folder))
# ll="libritts_testclean500_large"
ll="WavTokenizer_small_600_24k_4096"

tmptmp=out_folder+"/"+ll

os.system("rm -r %s"%(tmptmp))
os.system("mkdir -p %s"%(tmptmp))


def align_mse(orig, recon):
    # orig, recon: torch.Tensor channels x samples or 1D
    if orig.dim() > 1:
        orig = torch.mean(orig, dim=0)
    if recon.dim() > 1:
        recon = torch.mean(recon, dim=0)
    L = min(orig.shape[-1], recon.shape[-1])
    if L == 0:
        return None
    diff = orig[:L] - recon[:L]
    return float(torch.mean(diff ** 2).item())

# 自己数据模型加载
config_path = "../WavTokenizer_models/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
model_path = "../WavTokenizer_models/WavTokenizer_small_600_24k_4096.ckpt"
# config_path = "../WavTokenizer_models/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
# model_path = "../WavTokenizer_models/WavTokenizer_small_320_24k_4096.ckpt"
wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device1)
# wavtokenizer = wavtokenizer.to(device2)

with open(input_path,'r') as fin:
    x=fin.readlines()

x = [i.strip() for i in x]

# 完成一些加速处理

features_all=[]

for i in tqdm(range(len(x))):
    wav, sr = torchaudio.load(x[i])
    # print("***:",x[i])
    # wav = convert_audio(wav, sr, 24000, 1)                             # (1,131040)
    bandwidth_id = torch.tensor([0])
    wav=wav.to(device1)
    # print(i)

    features,discrete_code= wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
    features_all.append(features)

# wavtokenizer = wavtokenizer.to(device2)

for i in tqdm(range(len(x))):
    bandwidth_id = torch.tensor([0])
    bandwidth_id = bandwidth_id.to(device1)

    audio_out = wavtokenizer.decode(features_all[i], bandwidth_id=bandwidth_id)   
    # print(i,time.time()) 
    # breakpoint()                        # (1, 131200)
    audio_path = out_folder + '/' + ll + '/' + x[i].split('/')[-1]
    # os.makedirs(out_folder + '/' + ll, exist_ok=True)
    torchaudio.save(audio_path, audio_out.cpu(), sample_rate=24000, encoding='PCM_S', bits_per_sample=16)

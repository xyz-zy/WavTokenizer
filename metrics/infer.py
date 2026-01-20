# 测试各种指标
import os
import glob
import json
from UTMOS import UTMOSScore
from periodicity import calculate_periodicity_metrics
import torchaudio
from pesq import pesq
import numpy as np
import torch
import math
from pystoi import stoi
from tqdm import tqdm

device=torch.device('cuda:0')

# 如果是ljspeech，需要更换路径，更换数据读取逻辑，更换stoi的采样率

def main():
    prepath="./result/infer/WavTokenizer_small_600_24k_4096"
    rawpath="/home/lxz/orcd/pool/libritts/LibriTTS/test-clean"
    # rawpath="./Data/LJSpeech-1.1/wavs"
    preaudio = os.listdir(prepath)
    rawaudio = []

    UTMOS=UTMOSScore(device='cuda:0')
    
    # libritts
    for i in range(len(preaudio)):
        id1=preaudio[i].split('_')[0]
        id2=preaudio[i].split('_')[1]
        rawaudio.append(rawpath+"/"+id1+"/"+id2+"/"+preaudio[i])

    # # ljspeech
    # for i in range(len(preaudio)):
    #     rawaudio.append(rawpath+"/"+preaudio[i])

    utmos_sumgt=0.0
    utmos_sumencodec=0.0
    pesq_sumpre=0.0
    f1score_sumpre=0.0
    stoi_sumpre=[]
    f1score_filt=0

    per_file_metrics = {}

    bar = tqdm(preaudio, desc='Eval', unit='file')
    for i, preaudio_file in enumerate(bar):
        rawwav,rawwav_sr=torchaudio.load(rawaudio[i])
        prewav,prewav_sr=torchaudio.load(prepath+"/"+preaudio_file)
        # breakpoint()
        rawwav=rawwav.to(device)
        prewav=prewav.to(device)
        # print(rawwav.size(),prewav.size())
        # breakpoint()
        rawwav_16k=torchaudio.functional.resample(rawwav, orig_freq=rawwav_sr, new_freq=16000)  #测试UTMOS的时候必须重采样
        prewav_16k=torchaudio.functional.resample(prewav, orig_freq=prewav_sr, new_freq=16000)


        # 1.UTMOS
        utmos_raw = UTMOS.score(rawwav_16k.unsqueeze(1))[0].item()
        utmos_encodec = UTMOS.score(prewav_16k.unsqueeze(1))[0].item()
        # print("****UTMOS_raw", i, utmos_raw)
        # print("****UTMOS_encodec", i, utmos_encodec)
        utmos_sumgt += utmos_raw
        utmos_sumencodec += utmos_encodec
    

        # breakpoint()

        ## 2.PESQ  
        min_len=min(rawwav_16k.size()[1],prewav_16k.size()[1])
        rawwav_16k_pesq=rawwav_16k[:,:min_len].squeeze(0)
        prewav_16k_pesq=prewav_16k[:,:min_len].squeeze(0)
        try:
            pesq_score = pesq(16000, rawwav_16k_pesq.cpu().numpy(), prewav_16k_pesq.cpu().numpy(), "wb", on_error=1)
        except Exception:
            pesq_score = None
        # print("****PESQ", i, pesq_score)
        if pesq_score is not None:
            pesq_sumpre += pesq_score
        # breakpoint()

        ## 3.F1-score
        min_len=min(rawwav_16k.size()[1],prewav_16k.size()[1])
        rawwav_16k_f1score=rawwav_16k[:,:min_len]
        prewav_16k_f1score=prewav_16k[:,:min_len]
        periodicity_loss, pitch_loss, f1_score = calculate_periodicity_metrics(rawwav_16k_f1score, prewav_16k_f1score)
        # print("****f1", periodicity_loss, pitch_loss, f1_score)
        if math.isnan(f1_score):
            f1score_filt += 1
        else:
            f1score_sumpre += f1_score
        # breakpoint()


        ## 4.STOI
        # # 针对重采样的ljspeech
        # rawwav_24k=torchaudio.functional.resample(rawwav, orig_freq=rawwav_sr, new_freq=24000)
        # min_len=min(rawwav_24k.size()[1],prewav.size()[1])
        # rawwav_stoi=rawwav_24k[:,:min_len].squeeze(0)
        # prewav_stoi=prewav[:,:min_len].squeeze(0)
        # tmp_stoi=stoi(rawwav_stoi.cpu(),prewav_stoi.cpu(),24000,extended=False)
        # print("****stoi",tmp_stoi)
        # stoi_sumpre.append(tmp_stoi)
        # # breakpoint()

        # 针对libritts采样率是24k的
        min_len=min(rawwav.size()[1],prewav.size()[1])
        rawwav_stoi=rawwav[:,:min_len].squeeze(0)
        prewav_stoi=prewav[:,:min_len].squeeze(0)
        tmp_stoi = stoi(rawwav_stoi.cpu(), prewav_stoi.cpu(), rawwav_sr, extended=False)
        # print("****stoi", tmp_stoi)
        stoi_sumpre.append(tmp_stoi)

        # store per-file metrics
        fname = preaudio[i]
        per_file_metrics[fname] = {
            'UTMOS_raw': float(utmos_raw),
            'UTMOS_encodec': float(utmos_encodec),
            'PESQ': float(pesq_score) if pesq_score is not None else None,
            'periodicity_loss': float(periodicity_loss) if periodicity_loss is not None else None,
            'pitch_loss': float(pitch_loss) if pitch_loss is not None else None,
            'f1_score': float(f1_score) if not math.isnan(f1_score) else None,
            'STOI': float(tmp_stoi) if tmp_stoi is not None else None,
        }

        # update tqdm postfix with running averages
        processed = i + 1
        utmos_raw_mean = utmos_sumgt / processed if processed > 0 else None
        utmos_encodec_mean = utmos_sumencodec / processed if processed > 0 else None
        pesq_count = sum(1 for v in per_file_metrics.values() if v.get('PESQ') is not None)
        pesq_mean = (pesq_sumpre / pesq_count) if pesq_count > 0 else None
        valid_f1_count = (processed - f1score_filt)
        f1_mean = (f1score_sumpre / valid_f1_count) if valid_f1_count > 0 else None
        stoi_mean = float(np.mean(stoi_sumpre)) if len(stoi_sumpre) > 0 else None

        postfix = {
            'UTMOS_raw_mean': f"{utmos_raw_mean:.3f}" if utmos_raw_mean is not None else None,
            'UTMOS_encodec_mean': f"{utmos_encodec_mean:.3f}" if utmos_encodec_mean is not None else None,
            'PESQ_mean': f"{pesq_mean:.3f}" if pesq_mean is not None else None,
            'F1_mean': f"{f1_mean:.3f}" if f1_mean is not None else None,
            'STOI_mean': f"{stoi_mean:.3f}" if stoi_mean is not None else None,
        }
        bar.set_postfix(postfix)

    # compute summaries
    n = len(preaudio)
    summary = {}
    summary['UTMOS_raw_sum'] = float(utmos_sumgt)
    summary['UTMOS_raw_mean'] = float(utmos_sumgt / n) if n > 0 else None
    summary['UTMOS_encodec_sum'] = float(utmos_sumencodec)
    summary['UTMOS_encodec_mean'] = float(utmos_sumencodec / n) if n > 0 else None
    summary['PESQ_sum'] = float(pesq_sumpre)
    # PESQ mean: divide by number of non-None pesq entries
    pesq_count = sum(1 for v in per_file_metrics.values() if v.get('PESQ') is not None)
    summary['PESQ_mean'] = float(pesq_sumpre / pesq_count) if pesq_count > 0 else None
    # F1 mean over valid entries
    valid_f1_count = (n - f1score_filt)
    summary['F1_sum'] = float(f1score_sumpre)
    summary['F1_mean'] = float(f1score_sumpre / valid_f1_count) if valid_f1_count > 0 else None
    summary['F1_nan_count'] = int(f1score_filt)
    summary['STOI_mean'] = float(np.mean(stoi_sumpre)) if len(stoi_sumpre) > 0 else None

    # write per-file metrics and summary to json in prepath
    try:
        metrics_path = os.path.join(prepath, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(per_file_metrics, f, indent=2)
        summary_path = os.path.join(prepath, 'metrics_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print('Wrote metrics to', metrics_path, 'and', summary_path)
    except Exception as e:
        print('Failed to write metrics:', e)
    
    

if __name__=="__main__":
    main()
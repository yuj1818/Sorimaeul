'''
runtime\python.exe myinfer.py 0 "E:\codes\py39\RVC-beta\todo-songs\1111.wav" "E:\codes\py39\logs\mi-test\added_IVF677_Flat_nprobe_7.index" harvest "test.wav" "weights/mi-test.pth" 0.6 cuda:0 True
'''
import os,sys,pdb,torch
# now_dir = os.getcwd()
# sys.path.append(now_dir)
# import argparse
# import glob
import sys
import torch
# from multiprocessing import cpu_count
from infer.modules.vc.modules2 import VC
# from infer.lib.infer_pack.models import (
#     SynthesizerTrnMs256NSFsid,
#     SynthesizerTrnMs256NSFsid_nono,
#     SynthesizerTrnMs768NSFsid,
#     SynthesizerTrnMs768NSFsid_nono
# )
from infer.lib.audio import load_audio
# from fairseq import checkpoint_utils
from scipy.io import wavfile
from configs.config2 import Config
# from infer.modules.vc.pipeline import Pipeline

import logging

logger = logging.getLogger(__name__)

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
# os.environ["CUDA_VISIBLE_DEVICES"]= "9"  # Set the GPU 9 to use

# class Config:
#     def __init__(self,device,is_half):
#         self.device = device
#         self.is_half = is_half
#         self.n_cpu = 0
#         self.gpu_name = None
#         self.gpu_mem = None
#         self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

#     def device_config(self) -> tuple:
#         if torch.cuda.is_available():
#             i_device = int(self.device.split(":")[-1])
#             self.gpu_name = torch.cuda.get_device_name(i_device)
#             if (
#                 ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
#                 or "P40" in self.gpu_name.upper()
#                 or "1060" in self.gpu_name
#                 or "1070" in self.gpu_name
#                 or "1080" in self.gpu_name
#             ):
#                 print("16系/10系显卡和P40强制单精度")
#                 self.is_half = False
#                 for config_file in ["32k.json", "40k.json", "48k.json"]:
#                     with open(f"configs/{config_file}", "r") as f:
#                         strr = f.read().replace("true", "false")
#                     with open(f"configs/{config_file}", "w") as f:
#                         f.write(strr)
#                 with open("trainset_preprocess_pipeline_print.py", "r") as f:
#                     strr = f.read().replace("3.7", "3.0")
#                 with open("trainset_preprocess_pipeline_print.py", "w") as f:
#                     f.write(strr)
#             else:
#                 self.gpu_name = None
#             self.gpu_mem = int(
#                 torch.cuda.get_device_properties(i_device).total_memory
#                 / 1024
#                 / 1024
#                 / 1024
#                 + 0.4
#             )
#             if self.gpu_mem <= 4:
#                 with open("trainset_preprocess_pipeline_print.py", "r") as f:
#                     strr = f.read().replace("3.7", "3.0")
#                 with open("trainset_preprocess_pipeline_print.py", "w") as f:
#                     f.write(strr)
#         elif torch.backends.mps.is_available():
#             print("没有发现支持的N卡, 使用MPS进行推理")
#             self.device = "mps"
#         else:
#             print("没有发现支持的N卡, 使用CPU进行推理")
#             self.device = "cpu"
#             self.is_half = True

#         if self.n_cpu == 0:
#             self.n_cpu = cpu_count()

#         if self.is_half:
#             # 6G显存配置
#             x_pad = 3
#             x_query = 10
#             x_center = 60
#             x_max = 65
#         else:
#             # 5G显存配置
#             x_pad = 1
#             x_query = 6
#             x_center = 38
#             x_max = 41

#         if self.gpu_mem != None and self.gpu_mem <= 4:
#             x_pad = 1
#             x_query = 5
#             x_center = 30
#             x_max = 32

#         return x_pad, x_query, x_center, x_max

class infer:
    def __init__(self, f0up_key, input_path, index_path, f0method, opt_path, model_path, index_rate, device, is_half):
        self.f0up_key=f0up_key # 피치
        self.input_path=input_path # 노래 파일 위치
        self.index_path=index_path # 인덱스 파일 위치
        self.f0method=f0method # harvest or pm (알고리즘)
        self.opt_path=opt_path # 출력 파일 위치
        self.model_path=model_path # 모델 위치
        self.index_rate=index_rate # 인덱스 비율
        self.device=device # GPU 번호
        self.is_half=is_half # True
        self.config=Config(device)
        self.now_dir=os.getcwd()
        sys.path.append(self.now_dir)
        self.hubert_model=None
        self.cpt=None
        self.tgt_sr=None;

    # def load_hubert(self):
    #     global hubert_model
    #     models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(["hubert_base.pt"],suffix="",)
    #     hubert_model = models[0]
    #     hubert_model = hubert_model.to(self.device)
    #     if(self.is_half):hubert_model = hubert_model.half()
    #     else:hubert_model = hubert_model.float()
    #     hubert_model.eval()

    # def vc_single(self,sid,input_audio,f0_up_key,f0_file,f0_method,file_index,index_rate):
    #     global tgt_sr,net_g,vc,hubert_model,cpt
    #     if input_audio is None:return "You need to upload an audio", None
    #     audio=load_audio(input_audio,16000)
    #     times = [0, 0, 0]
    #     if(hubert_model==None):self.load_hubert()
    #     if_f0 = cpt.get("f0", 1)
    #     # audio_opt=vc.pipeline(hubert_model,net_g,sid,audio,times,f0_up_key,f0_method,file_index,file_big_npy,index_rate,if_f0,f0_file=f0_file)
    #     audio_opt=vc.pipeline(hubert_model,net_g,sid,audio,times,f0_up_key,f0_method,file_index,index_rate,if_f0,f0_file=f0_file)
    #     print(times)
    #     return audio_opt


    # def get_vc(self,model_path):
    #     print("loading pth %s"%model_path)
    #     self.cpt = torch.load(model_path, map_location="cpu")
    #     self.tgt_sr = self.cpt["config"][-1]
    #     self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    #     self.if_f0 = self.cpt.get("f0", 1)
    #     self.version = self.cpt.get("version", "v1")

    #     synthesizer_class = {
    #         ("v1", 1): SynthesizerTrnMs256NSFsid,
    #         ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
    #         ("v2", 1): SynthesizerTrnMs768NSFsid,
    #         ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
    #     }

    #     self.net_g = synthesizer_class.get(
    #         (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
    #     )(*self.cpt["config"], is_half=self.config.is_half)

    #     del self.net_g.enc_q

    #     self.net_g.load_state_dict(self.cpt["weight"], strict=False)
    #     self.net_g.eval().to(self.config.device)
    #     if self.config.is_half:
    #         self.net_g = self.net_g.half()
    #     else:
    #         self.net_g = self.net_g.float()

    #     self.pipeline = Pipeline(self.tgt_sr, self.config)
    #     n_spk = self.cpt["config"][-3]

    #     return {"visible": True, "maximum": n_spk, "__type__": "update"}

    def run(self):
        # vc = self.get_vc(model_path=self.model_path)
        vc = VC(self.config)
        vc.get_vc("pth.pth", self.model_path, self.index_path, 0.33, 0.33)

        msg, (tgt_sr, audio_opt) = vc.vc_single(0,
                            self.input_path,
                            self.f0up_key,
                            self.model_path,
                            self.f0method,
                            self.index_path,
                            self.index_path,
                            self.index_rate,
                            3,
                            0,
                            0.25,
                            0.33)
        # print(f"audio_opt: {audio_opt}")

        logger.info(msg)
        logger.info(f"Write file {self.opt_path}")

        wavfile.write(self.opt_path, tgt_sr, audio_opt)


import myinfer as mif

import os

import logging

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "9"  # Set the GPU 9 to use

logger = logging.getLogger(__name__)

cur_dir = os.getcwd()
dubbing_path = f"{cur_dir}/dubbing"
model_path = f"{cur_dir}/model"

class Infer:

    def infer(pitch, modelCode, file_path, file_name):
        origin_file = f"{file_path}/{file_name}"

        pth = f"{model_path}/{modelCode}/pth.pth"
        index = f"{model_path}/{modelCode}/index.index"
        index = None

        inferred = f"{file_path}/inferred.wav"

        logger.info("Start infer")

        inf = mif.infer(f0up_key=pitch,
                        input_path=origin_file,
                        index_path=index,
                        f0method="rmvpe",
                        opt_path=inferred,
                        model_path=pth,
                        index_rate=0.6,
                        device="cuda:0",
                        is_half=True)
        
        inf.run()

        return inferred

##Step1
#########################################################################
##### Please run on KneronToolchain Base Env. (conda activate base) #####
#########################################################################
# #On KneronToolchain (Base Env.)
# m = onnx.load("/mnt/models/yolox_wade_194_torch17.onnx") ## Load your onnx model   
# m_opt = ktc.onnx_optimizer.onnx2onnx_flow(m) ## Please use base's env. ktc.onnx_optimizer.onnx2onnx_flow to optimze model, this step can del cpu nodes (e.g. slice)
# onnx.save(m_opt, "/mnt/models/yolox_wade_194_torch17_opt.onnx") ## Please make sure there are no cpu nodes (e.g. slice) in front of the onnx model.
# exit()

##Step2
#################################################################################
##### Please run on KneronToolchain onnx1.13 Env. (conda activate onnx1.13) #####
#################################################################################
#On KneronToolchain (onnx1.13 Env.)

import argparse  
import ktc
import numpy as np
import os
import onnx
from PIL import Image
import random

save_path = '/mnt/models/yolox_wade_194_torch17/'
onnx_path = '/mnt/models/yolox_wade_194_torch17.onnx'
sample_image_path = "/mnt/dataset2/val"

parser = argparse.ArgumentParser(description="Run optimization and simulation in different environments")  
parser.add_argument('--step', type=int, choices=[1, 2], required=True, help="Step to run (1 or 2)")  
parser.add_argument('--onnx_path', type=str, default=onnx_path, required=False, help="Path to the ONNX model")  
parser.add_argument('--sample_image_path', type=str, default=sample_image_path, required=False, help="Path to sample images (required for step 2)")  
parser.add_argument('--save_path', type=str, default=save_path, required=False, help="Path to save the results (required for step 2)")  

args = parser.parse_args()  

### This only work in (base evc) "docker exec -it kntoolchain bash -> conda activate base"
############################################################

if args.step == 1:
    m = onnx.load(onnx_path)
    m = ktc.onnx_optimizer.onnx2onnx_flow(m)
    onnx.save(m, onnx_path[:-5]+'_opt.onnx')
    exit()

############################################################

### This only work in (onnx1.13 evc) "docker exec -it kntoolchain bash"
############################################################
elif args.step == 2:
    # npu (only) performance simulation
    m_opt = onnx.load(onnx_path[:-5]+'_opt.onnx')
    km = ktc.ModelConfig(20008, "0001", "720", onnx_model=m_opt)
    eval_result = km.evaluate(output_dir=save_path)
    print("\nNpu performance evaluation result:\n" + str(eval_result))

    import os
    from os import walk

    img_list = []
    for (dirpath, dirnames, filenames) in walk(sample_image_path):
        for f in filenames:
            fullpath = os.path.join(dirpath, f)
            try:  
                image = Image.open(fullpath)  
                image = image.convert("RGB")  
                image = Image.fromarray(np.array(image)[..., ::-1])  
                img_data = np.array(image.resize((640, 640), Image.BILINEAR)) / 256 - 0.5  
                img_data = np.expand_dims(img_data, axis=0)  
                img_data = np.transpose(img_data, (0, 3, 1, 2))  
                print(fullpath)  
                img_list.append(img_data)  
            except Exception as e:  
                print(f"Error processing image {fullpath}: {e}") 

    # 隨機選取 200 張圖片  
    max_images = 200  
    if len(img_list) > max_images:  
        img_list = random.sample(img_list, max_images)  
        
    # fixed-point analysis
    # bie_model_path = km.analysis({"input": img_list}, output_dir=save_path, threads=3)
    # 可選更精細量化的方法，但需要更久時間
    bie_model_path = km.analysis({"input": img_list}, output_dir=save_path, threads=3, mode=3, optimize=2)
    print("\nFixed-point analysis done. Saved bie model to '" + str(bie_model_path) + "'")

    # compile
    nef_model_path = ktc.compile([km])
    print("\nCompile done. Saved Nef file to '" + str(nef_model_path) + "'")

############################################################

# all nef file will save at /data1/kneron_flow


# Import necessary libraries for image processing.
from PIL import Image
import numpy as np
import ktc
import time

# A very simple preprocess function. Note that the image should be in the same format as the ONNX input (usually NCHW).
def preprocess(input_file):
    image = Image.open(input_file)
    image = image.convert("RGB")
    img_data = np.array(image.resize((640, 640), Image.BILINEAR)) / 255
    img_data = np.transpose(img_data, (2, 0, 1))
    # Expand 3x224x224 to 1x3x224x224 which is the shape of ONNX input.
    img_data = np.expand_dims(img_data, 0)
    print(img_data.shape)
    return img_data

# Use the previous function to preprocess an example image as the input.
input_data = [preprocess("/mnt/dataset/val/Capture_20241111_6.mp4_105.jpg")]

# The `onnx_file` is generated and saved in the previous code block.
# The `input_names` are the input names of the model.
# The `input_data` order should be kept corresponding to the input names. It should be in the same shape as ONNX (NCHW).
# The inference result will be save as a list of array.

onnx_s = time.time()
onnx_path = '/mnt/models/kneron_yolox_wade_epoch119_best.onnx'
floating_point_inf_results = ktc.kneron_inference(input_data, onnx_file=onnx_path, input_names=["input"])
print("onnx: ", floating_point_inf_results.shape)
onnx_e = time.time()

# bie_s = time.time()
# bie_path = '/mnt/models/kneron_yolox_119_best/input.kdp720.scaled.bie'
# fixed_point_inf_results = ktc.kneron_inference(input_data, bie_file=bie_path, input_names=["input"], platform=720)
# print("bie: ", fixed_point_inf_results)
# bie_e = time.time()

# nef_s = time.time()
# nef_path = '/mnt/models/kneron_yolox_119_best/models_720.nef'
# fixed_point_inf_results = ktc.kneron_inference(input_data, nef_file=nef_path, input_names=["input"], platform=720)
# print("nef: ", fixed_point_inf_results)
# nef_e = time.time()

print('onnx_time: ', onnx_e - onnx_s)
# print('bie_time: ', bie_e - bie_s)
# print('nef_time: ', nef_e - nef_s)

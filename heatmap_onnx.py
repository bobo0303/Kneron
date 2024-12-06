# Import necessary libraries for image processing.
from PIL import Image
import numpy as np
import ktc
import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def min_max_norm(x):
    x_min = np.min(x)  
    x_max = np.max(x)  
    x_normalized = (x - x_min) / (x_max - x_min)  
    return x_normalized

def create_colormap(threshold):
    # Create a colormap with two gradients: blue to white for values below the threshold, red to black for values above
    colors_below = plt.cm.Blues(np.linspace(0, threshold, 128))
    colors_above = plt.cm.Reds(np.linspace(threshold, 1, 128))

    # Combine the two colormaps
    colors = np.vstack((colors_below, colors_above))
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Create a normalization object that maps the threshold to the middle of the colormap
    boundaries = np.concatenate(
        [np.linspace(0, threshold, 128), np.linspace(threshold, 1, 128)]
    )
    norm = mcolors.BoundaryNorm(boundaries, ncolors=256)

    return cmap, norm


def show_heatmap(feature_map, threshold, name):
    fm_80 = feature_map[0].copy()
    fm_40 = feature_map[1].copy()
    fm_20 = feature_map[2].copy()

    # Create colormap and normalization
    cmap, norm = create_colormap(threshold)

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot each tensor in a subplot
    for ax, tensor, title in zip(
        axes, [fm_80, fm_40, fm_20], ["80x80", "40x40", "20x20"]
    ):
        tensor_sigmoid = min_max_norm(tensor)
        tensor_sigmoid = tensor_sigmoid[
            0, 0, :, :
        ]  # Assuming the shape is (batch, channel, height, width)
        im = ax.imshow(tensor_sigmoid, cmap=cmap, norm=norm)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    # Save the combined figure
    plt.savefig("combined_" + name)
    plt.close()

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

image_name = "/mnt/images/rf_image_97.jpg"
# Use the previous function to preprocess an example image as the input.
input_data = [preprocess(image_name)]

# The `onnx_file` is generated and saved in the previous code block.
# The `input_names` are the input names of the model.
# The `input_data` order should be kept corresponding to the input names. It should be in the same shape as ONNX (NCHW).
# The inference result will be save as a list of array.

onnx_s = time.time()
onnx_path = '/mnt/models/kneron_yolox_wade_epoch119_best.onnx'
floating_point_inf_results = ktc.kneron_inference(input_data, onnx_file=onnx_path, input_names=["input"])
# print("onnx: ", floating_point_inf_results)

print("onnx_obj: ", floating_point_inf_results[-3].shape)
print("onnx_obj: ", floating_point_inf_results[-2].shape)
print("onnx_obj: ", floating_point_inf_results[-1].shape)
obj_check = floating_point_inf_results[-3:]  # obj 80-40-20
show_heatmap(obj_check, 0.6, image_name.split('/')[-1])
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

# Import necessary libraries for image processing.
import os
import sys
import argparse
import cv2

from PIL import Image
import numpy as np
import ktc
import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np
import math

######################################### KL720KnModelZooGenericDataInferenceMMDetYoloX.py function ##########################################

def decode_outputs(outputs_np, hw_np, layer_strides=[8, 16, 32]):
    """decode outputs for a batch item into bbox predictions.
    Args:
        outputs_np: The anchor aggregated output, with last channel in reg(4), obj_score(1), cls_score(80), 4+1+80 = 85
                    [batch, n_anchors_all, 85] = [batch, 80x80 + 40x40 + 20x20, 85] = [1,8400,85]
        hw_np: the feature_size of the three PAN output layers, example: [(80, 80), (40, 40), (20, 20)] for 640x640
    Returns:
        The decoded: (Batch, anchor, 7): The last dimension is [cx, cy, w, h, obj_score, cls_score, label]
    """

    grids_np = []
    strides_np = []
    for (hsize, wsize), stride in zip(hw_np, layer_strides):
        """
        # https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
        Giving the string ‘ij’ returns a meshgrid with matrix indexing,
        while ‘xy’ returns a meshgrid with Cartesian indexing.
        In the 2-D case with inputs of length M and N,
        the outputs are of shape (N, M) for ‘xy’ indexing and (M, N) for ‘ij’ indexing
        """
        yv, xv = np.meshgrid(np.arange(0, hsize), np.arange(0, wsize), indexing="ij")
        grid_np = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids_np.append(grid_np)
        shape = grid_np.shape[:2]
        strides_np.append(np.full((*shape, 1), stride, dtype=np.float32))

    grids_np = np.concatenate(grids_np, axis=1).astype(np.float32)
    strides_np = np.concatenate(strides_np, axis=1).astype(np.float32)

    outputs_np[..., :2] = (outputs_np[..., :2] + grids_np) * strides_np  # center
    outputs_np[..., 2:4] = np.exp(outputs_np[..., 2:4]) * strides_np  # bbox size

    return outputs_np


def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, a_min=0.0, a_max=None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])

    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold=0.5, top_k=-1, candidate_size=200):
    """
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []

    indexes = np.argsort(scores)[::-1]

    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(rest_boxes, np.expand_dims(current_box, 0))
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def yolox_postprocess(
    prediction_np, num_classes, prob_threshold=0.3, iou_threshold=0.5, topk=300
):
    """
    Args:
        prediction_np: (Batch, anchor, 7): The last dimension is [cx, cy, w, h, obj_score, cls_score, label]
        num_classes: number of classes, for COCO = 80
        prob_threshold: threshold of scores filtering of object
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        box_probs: (N, 6): The detected objects, The last dimension is [x1, y1, x2, y2, score, label]
    """
    box_corner_np = np.zeros_like(prediction_np)
    box_corner_np[:, :, 0] = prediction_np[:, :, 0] - prediction_np[:, :, 2] / 2
    box_corner_np[:, :, 1] = prediction_np[:, :, 1] - prediction_np[:, :, 3] / 2
    box_corner_np[:, :, 2] = prediction_np[:, :, 0] + prediction_np[:, :, 2] / 2
    box_corner_np[:, :, 3] = prediction_np[:, :, 1] + prediction_np[:, :, 3] / 2
    prediction_np[:, :, :4] = box_corner_np[:, :, :4]

    # compute the argmax of all (COCO=80) classes and check if pass the threshold
    image_pred_np = prediction_np[0]
    class_conf_np = np.amax(image_pred_np[:, 5 : 5 + num_classes], 1, keepdims=True)
    class_pred_np = np.argmax(image_pred_np[:, 5 : 5 + num_classes], 1)
    class_pred_np = np.expand_dims(class_pred_np, -1)
    class_pred_np = class_pred_np.astype(np.float32)
    conf_mask_np = np.squeeze(
        image_pred_np[:, 4] * np.squeeze(class_conf_np) >= prob_threshold
    )

    # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    detections_np = np.concatenate(
        (image_pred_np[:, :5], class_conf_np, class_pred_np), 1
    )
    # x1 y1 x2 y2 obj_conf cls_conf, label
    detections_np = detections_np[conf_mask_np]

    box_probs = np.empty((0, 6), dtype=float)
    if not detections_np.shape[0]:
        return box_probs

    # class-wised NMS
    for class_index in range(num_classes):
        mask = np.squeeze(detections_np[:, 6] == class_index)
        cls_score = detections_np[:, 5]
        cls_score = cls_score[mask].reshape(-1, 1)
        if cls_score.shape[0] == 0:
            continue
        reg_bbox = detections_np[:, :4]
        reg_bbox = reg_bbox[mask]
        obj_score = detections_np[:, 4]
        obj_score = obj_score[mask].reshape(-1, 1)
        box_probs_class = np.concatenate([reg_bbox, obj_score * cls_score], axis=1)

        box_probs_class = hard_nms(box_probs_class, iou_threshold, topk)
        label = np.ones((box_probs_class.shape[0], 1)) * (
            class_index
        )  # adding label to box_prob
        box_probs_class = np.concatenate([box_probs_class, label], axis=1)
        box_probs = np.concatenate([box_probs, box_probs_class], axis=0)
    return box_probs


def get_bboxes(
    outputs_np,
    max_shape,
    num_classes=10,
    prob_threshold=0.1,
    iou_threshold=0.5,
    topk=300,
):
    """Transform network output for a batch into bbox predictions.
    Args:
        outputs_np : a list of three PAN output layers, of reg_out, obj_out, cls_out
                    for 640x640. outpus_npshape  is with shape [ (1, 80, 80, 4), (1, 80, 80, 1)  (1, 80, 80, 80) ],
                                                                 [(1, 40, 40, 4), (1, 40, 40, 1), (1, 40, 40, 80) ],
                                                                 [(1, 20, 20, 4), (1, 20, 20, 1), (1, 20, 20, 80) ]
        max_shape: H,W onnx input size
        num_classes: number of classes, for COCO = 80
        prob_threshold: threshold of scores filtering of object
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        dets: (N, 6): The detected objects, The last dimension is [x1, y1, x2, y2, score, label]

    """
    # for 640x640. outpus_np shape is with shape (1, 80, 80, 85), (1, 40, 40, 85),(1, 20, 20, 85)
    # for 488x800 and 1 class: (1, 56, 100, 6), (1, 28, 50, 6), (1, 14, 25, 6)
    outputs_np = [
        np.concatenate([reg_out, sigmoid(obj_out), sigmoid(cls_out)], axis=3)
        for reg_out, obj_out, cls_out in outputs_np
    ]

    # for 640x640 hw_np is with shape[(80, 80), (40, 40), (20, 20)]
    # for 488x800 ahw_np is with shape [(56, 100), (28, 50), (14, 25)]
    hw_np = [x.shape[1:3] for x in outputs_np]
    # [batch, n_anchors_all, 85] = [batch, 80x80 + 40x40 + 20x20, 85] = [1,8400,85]
    # =============== aggregate anchor start =================================
    for i in range(len(outputs_np)):
        batch, feature_h, feature_w, channel = outputs_np[i].shape
        outputs_np[i] = outputs_np[i].reshape(batch, -1, channel)
    outputs_np = np.concatenate([outputs_np[0], outputs_np[1], outputs_np[2]], axis=1)
    # outputs_np = np.transpose(outputs_np, (0,2,1))
    # =============== aggregate anchor end =================================

    prediction_np = decode_outputs(outputs_np, hw_np)
    dets = yolox_postprocess(
        prediction_np, num_classes, prob_threshold, iou_threshold, topk
    )

    for det in dets:
        det[0] = np.clip(det[0], 0, max_shape[1] - 1)
        det[1] = np.clip(det[1], 0, max_shape[0] - 1)
        det[2] = np.clip(det[2], 0, max_shape[1] - 1)
        det[3] = np.clip(det[3], 0, max_shape[0] - 1)
    return dets


def postprocess_(inf_results, **kwargs):
    """
    Input:
        outputs_np : a list of three PAN output layers, of reg_out, obj_out, cls_out
                    for 640x640. outpus_npshape  is with shape [ (1, 80, 80, 4), (1, 80, 80, 1)  (1, 80, 80, 11) ,
                                                                 (1, 40, 40, 4), (1, 40, 40, 1), (1, 40, 40, 11) ,
                                                                 (1, 20, 20, 4), (1, 20, 20, 1), (1, 20, 20, 11) ]
        input_shape: onnx input shape
        num_classes: number of classes
        conf_thres: confidence threshold
        iou_thres: iou threshold

    """
    num_pan_layer = 3
    outputs_np = []
    for i in range(num_pan_layer):
        reg_out = np.asarray(inf_results[3 * i + 0])
        obj_out = np.asarray(inf_results[3 * i + 1])
        cls_out = np.asarray(inf_results[3 * i + 2])
        assert (
            reg_out.shape[3] > obj_out.shape[3]
        ), f"Regression node channel length smaller than objectness node channel length(reg: {reg_out.shape[3]} v.s. obj: {obj_out.shape[3]}), which is not allow in this postprocess function. Make sure your input array is in correct oreder."
        outputs_np.append([reg_out, obj_out, cls_out])

    input_shape = (kwargs["input_shape"][0], kwargs["input_shape"][1])

    # get the bbox from anchor information
    box_probs = get_bboxes(
        outputs_np=outputs_np,
        max_shape=input_shape,
        num_classes=kwargs["num_classes"],
        prob_threshold=kwargs["conf_threshold"],
        iou_threshold=kwargs["iou_threshold"],
        topk=kwargs["top_k_num"],
    )

    # scale back to original image
    for i in range(len(box_probs)):
        box_probs[i, 2:4] = box_probs[i, 2:4] - box_probs[i, :2]

    return box_probs.tolist()


def convert_numpy_to_rgba_and_width_align_4(data):
    """Converts the numpy data into RGBA.

    720 input is 4 byte width aligned.

    """

    height, width, channel = data.shape

    width_aligned = 4 * math.ceil(width / 4.0)
    aligned_data = np.zeros((height, width_aligned, 4), dtype=np.int8)
    aligned_data[:height, :width, :channel] = data
    aligned_data = aligned_data.flatten()

    return aligned_data.tobytes()

def post_process(inf_node_output_list):
    """
    post-process the last raw output
    """

    tmp = []
    for o_n in inf_node_output_list:
        o_array = o_n.copy()
        o_array = o_array.transpose((0, 2, 3, 1))
        tmp.append(o_array)

    out_data = [tmp[3], tmp[6], tmp[0], tmp[4], tmp[7], tmp[1], tmp[5], tmp[8], tmp[2]]

    kwargs = {
        "input_shape": [640, 640],
        # "input_shape": [nef_model_width, nef_model_height],
        "num_classes": 11,
        "conf_threshold": 0.25,
        "iou_threshold": 0.5,
        "top_k_num": 20,
    }
    det_res = postprocess_(out_data, **kwargs)
    
    return det_res
    
def output_result_image(image_file_path, det_res, types):
    """
    output result image
    """

    print("[Output Result Image]")
    output_img_name = f"/mnt/output_{os.path.basename(image_file_path[:-4])}_{types}.jpg"
    print(" - Output bounding boxes on '{}'".format(output_img_name))
    img = cv2.imread(image_file_path)
    for bbox in det_res:
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = xmin + bbox[2]
        ymax = ymin + bbox[3]
        score = bbox[4]
        label = int(bbox[5])  # 類別標籤

        xmin = xmin * (img_width / 640)
        xmax = xmax * (img_width / 640)
        ymin = ymin * (img_height / 640)
        ymax = ymax * (img_height / 640)
        cv2.rectangle(
            img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2
        )
        text = f"Class: {label}, Confidence: {score:.2f}"
        cv2.putText(
            img,
            text,
            (int(xmin), int(ymin) + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        print(
            "("
            + str(int(xmin))
            + ","
            + str(int(ymin))
            + ","
            + str(int(xmax))
            + ","
            + str(int(ymax))
            + ")"
        )
    cv2.imwrite(output_img_name, img=img)


######################################### KL720KnModelZooGenericDataInferenceMMDetYoloX.py function ##########################################

######################################################### heatmap function ##########################################################

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
    print("[Output heatmap Image]")
    # Save the combined figure
    plt.savefig("/mnt/combined_" + name)
    print(f" - Output heatmap on 'combined_{name}'")
    
    plt.close()

######################################################### heatmap function ##########################################################

################################################### ktc.kneron_inference function ###################################################

# A very simple preprocess function. Note that the image should be in the same format as the ONNX input (usually NCHW).
def preprocess(input_file):
    image = Image.open(input_file)
    image = image.convert("RGB")
    img_data = np.array(image.resize((640, 640), Image.BILINEAR)) / 256
    img_data -= 0.5
    img_data = np.transpose(img_data, (2, 0, 1))
    # Expand 3x224x224 to 1x3x224x224 which is the shape of ONNX input.
    img_data = np.expand_dims(img_data, 0)
    print(img_data.shape)
    return img_data

################################################### ktc.kneron_inference function ###################################################


image_name = "/mnt/dataset2/test/WIN_20241119_10_03_47_Pro_297.jpg"
# Use the previous function to preprocess an example image as the input.
img = cv2.imread(filename=image_name)
img_height, img_width, img_channels = img.shape
input_data = [preprocess(image_name)]

onnx_s = time.time()
onnx_path = '/mnt/models/yolox_wade_194_torch17.onnx'
floating_point_inf_results = ktc.kneron_inference(input_data, onnx_file=onnx_path, input_names=["input"])
# print("onnx: ", floating_point_inf_results)
obj_check = floating_point_inf_results[-3:]  # obj 80-40-20
show_heatmap(obj_check, 0.6, image_name.split('/')[-1][:-4]+"_onnx.jpg")

det_res = post_process(floating_point_inf_results)
output_result_image(image_name, det_res, "onnx")
onnx_e = time.time()


nef_s = time.time()
nef_path = '/mnt/models/yolox_wade_194_torch17/models_720.nef'
fixed_point_inf_results = ktc.kneron_inference(input_data, nef_file=nef_path, input_names=["input"], platform=720)
# print("nef: ", fixed_point_inf_results)
obj_check = fixed_point_inf_results[-3:]  # obj 80-40-20
show_heatmap(obj_check, 0.6, image_name.split('/')[-1][:-4]+"_nef.jpg")

det_res = post_process(fixed_point_inf_results)
output_result_image(image_name, det_res, "nef")
nef_e = time.time()

print('onnx_time: ', onnx_e - onnx_s)
print('nef_time: ', nef_e - nef_s)
    
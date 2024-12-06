# ******************************************************************************
#  Copyright (c) 2021-2022. Kneron Inc. All rights reserved.                   *
# ******************************************************************************

import argparse
import os
import platform
import sys
import threading
import time

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(PWD, '..'))

from utils.ExampleHelper import get_device_usb_speed_by_port_id
from utils.ExamplePostProcess import post_process_yolo_v5
import kp
import cv2

MODEL_FILE_PATH = os.path.join(PWD, '../../res/models/KL720/YoloV5s_640_640_3/models_720.nef')

_LOCK = threading.Lock()
_SEND_RUNNING = True
_RECEIVE_RUNNING = True

_image_to_inference = None
_image_to_show = None

_generic_inference_input_descriptor = kp.GenericImageInferenceDescriptor()


def _image_send_function(_device_group: kp.DeviceGroup) -> None:
    global _image_to_inference, _generic_inference_input_descriptor, _SEND_RUNNING, _RECEIVE_RUNNING

    try:
        # set camera configuration
        if 'Windows' == platform.system():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        try_cap_time = 0

        while _SEND_RUNNING:
            try:
                # read frame from camera
                if cap.isOpened():
                    ret, _image_to_inference = cap.read()

                    if not ret and try_cap_time < 50:
                        try_cap_time += 1
                        time.sleep(0.1)
                        continue
                    elif not ret:
                        print('Error: opencv open camera failed')
                        break
                    else:
                        try_cap_time = 0
                else:
                    print('Error: opencv open camera failed')
                    break
            except:
                print('Error: opencv read frame failed')
                break

            # convert color space to RGB565
            inference_image = cv2.cvtColor(src=_image_to_inference, code=cv2.COLOR_BGR2BGR565)

            _generic_inference_input_descriptor.input_node_image_list = [
                kp.GenericInputNodeImage(
                    image=inference_image,
                    resize_mode=kp.ResizeMode.KP_RESIZE_ENABLE,
                    padding_mode=kp.PaddingMode.KP_PADDING_CORNER,
                    normalize_mode=kp.NormalizeMode.KP_NORMALIZE_KNERON
                )
            ]

            # inference image
            if ret:
                kp.inference.generic_image_inference_send(device_group=_device_group,
                                                          generic_inference_input_descriptor=_generic_inference_input_descriptor)

        _RECEIVE_RUNNING = False
        cap.release()
    except kp.ApiKPException as exception:
        print(' - Error: inference failed, error = {}'.format(exception))

    _SEND_RUNNING = False
    _RECEIVE_RUNNING = False


def _result_receive_function(_device_group: kp.DeviceGroup) -> None:
    global _image_to_show, _RECEIVE_RUNNING
    _loop = 0
    _fps = 0

    while _RECEIVE_RUNNING:
        time_start = time.time()

        # receive inference result
        try:
            generic_raw_result = kp.inference.generic_image_inference_receive(device_group=_device_group)
        except kp.ApiKPException as exception:
            print(' - Error: inference failed, error = {}'.format(exception))
            exit(0)

        _loop += 1

        with _LOCK:
            temp_image = _image_to_inference.copy()

        # retrieve inference node output
        inf_node_output_list = []
        for node_idx in range(generic_raw_result.header.num_output_node):
            inference_float_node_output = kp.inference.generic_inference_retrieve_float_node(node_idx=node_idx,
                                                                                             generic_raw_result=generic_raw_result,
                                                                                             channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW)
            inf_node_output_list.append(inference_float_node_output)

        # do post-process
        _yolo_result = post_process_yolo_v5(inference_float_node_output_list=inf_node_output_list,
                                            hardware_preproc_info=generic_raw_result.header.hw_pre_proc_info_list[0],
                                            thresh_value=0.15,
                                            with_sigmoid=False)

        # draw bounding box
        for yolo_result in _yolo_result.box_list:
            b = 100 + (25 * yolo_result.class_num) % 156
            g = 100 + (80 + 40 * yolo_result.class_num) % 156
            r = 100 + (120 + 60 * yolo_result.class_num) % 156
            color = (b, g, r)

            cv2.rectangle(img=temp_image,
                          pt1=(int(yolo_result.x1), int(yolo_result.y1)),
                          pt2=(int(yolo_result.x2), int(yolo_result.y2)),
                          color=color,
                          thickness=3)

        time_end = time.time()

        if 30 == _loop:
            _fps = 1 / (time_end - time_start)
            _loop = 0

        # draw FPS
        cv2.putText(img=temp_image,
                    text='FPS: {:.2f}'.format(_fps),
                    org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1,
                    color=(200, 200, 200),
                    thickness=1,
                    lineType=cv2.LINE_AA)
        cv2.putText(img=temp_image,
                    text='Press \'ESC\' to exit',
                    org=(10, temp_image.shape[0] - 10),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1,
                    color=(200, 200, 200),
                    thickness=1,
                    lineType=cv2.LINE_AA)

        with _LOCK:
            _image_to_show = temp_image.copy()

    _RECEIVE_RUNNING = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KL720 Demo Camera Generic Image Inference Example.')
    parser.add_argument('-p',
                        '--port_id',
                        help='Using specified port ID for connecting device (Default: port ID of first scanned Kneron '
                             'device)',
                        default=0,
                        type=int)
    args = parser.parse_args()

    usb_port_id = args.port_id

    """
    check device USB speed (Recommend run KL720 at super speed)
    """
    try:
        if kp.UsbSpeed.KP_USB_SPEED_SUPER != get_device_usb_speed_by_port_id(usb_port_id=usb_port_id):
            print('\033[91m' + '[Error] Device is not run at super speed.' + '\033[0m')
            exit(0)
    except Exception as exception:
        print('Error: check device USB speed fail, port ID = \'{}\', error msg: [{}]'.format(usb_port_id,
                                                                                             str(exception)))
        exit(0)

    """
    connect the device
    """
    try:
        print('[Connect Device]')
        device_group = kp.core.connect_devices(usb_port_ids=[usb_port_id])
        print(' - Success')
    except kp.ApiKPException as exception:
        print('Error: connect device fail, port ID = \'{}\', error msg: [{}]'.format(usb_port_id,
                                                                                     str(exception)))
        exit(0)

    """
    setting timeout of the usb communication with the device
    """
    print('[Set Device Timeout]')
    kp.core.set_timeout(device_group=device_group, milliseconds=10000)
    print(' - Success')

    """
    upload model to device
    """
    try:
        print('[Upload Model]')
        model_nef_descriptor = kp.core.load_model_from_file(device_group=device_group,
                                                            file_path=MODEL_FILE_PATH)
        print(' - Success')

        print('[Model NEF Information]')
        print(model_nef_descriptor)
    except kp.ApiKPException as exception:
        print('Error: upload model failed, error = \'{}\''.format(str(exception)))
        exit(0)

    """
    prepare generic image inference input descriptor
    """
    _generic_inference_input_descriptor = kp.GenericImageInferenceDescriptor(
        model_id=model_nef_descriptor.models[0].id,
        inference_number=0,
    )

    """
    starting inference work
    """
    print('[Starting Inference Work]')
    print(' - Starting inference')
    send_thread = threading.Thread(target=_image_send_function, args=(device_group,))
    receive_thread = threading.Thread(target=_result_receive_function, args=(device_group,))

    send_thread.start()
    receive_thread.start()

    cv2.namedWindow('Generic Inference', cv2.WND_PROP_ASPECT_RATIO or cv2.WINDOW_GUI_EXPANDED)
    cv2.setWindowProperty('Generic Inference', cv2.WND_PROP_ASPECT_RATIO, cv2.WND_PROP_ASPECT_RATIO)

    try:
        while True:
            with _LOCK:
                if None is not _image_to_show:
                    cv2.imshow('Generic Inference', _image_to_show)

            if (27 == cv2.waitKey(10)) or (not _SEND_RUNNING) or (not _RECEIVE_RUNNING):
                break
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()

    _SEND_RUNNING = False

    send_thread.join()
    receive_thread.join()

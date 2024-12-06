"""
Generic inference function for ONNX, BIE, or NEF model.
"""
import pathlib
import sys
from typing import List, Optional

import numpy.typing as npt
from sys_flow.compiler_v2 import unpack_nefs as unpack_nefs_v1
from sys_flow.inference import get_model_io as get_model_io_v1
from sys_flow_v2.compiler_v2 import unpack_nefs as unpack_nefs_v2
from sys_flow_v2.inference import get_model_io as get_model_io_v2

ROOT_FOLDER = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_FOLDER))

from python_flow.common import constants
from python_flow.common import exceptions
from python_flow.utils import csim
from python_flow.utils import dynasty
from python_flow.utils import utils

def kneron_inference(pre_results: List[npt.ArrayLike], input_names: Optional[List[str]],
                     nef_file: str = "", onnx_file: str = "", bie_file: str = "",
                     model_id: Optional[int] = None, data_type: str = "float",
                     reordering: Optional[List[str]] = None,
                     platform: int = 520) -> List[npt.ArrayLike]:
    """Performs inference on the input model given the specified parameters.

    Input pre_results should be in channel first.

    Arguments:
        pre_results: List of NumPy arrays in channel first format from preprocessing.
        input_names: List of input node names of the model.
        nef_file: String path to NEF model for inference.
        onnx_file: String path to ONNX model for inference, unused if nef_file is specified.
        bie_file: String path to BIE model for inference, unused if nef_file/onnx_file is
            specified.
        model_id: Integer of model to run inference, only necessary for NEF with multiple models.
        data_type: String format of the resulting output, "fixed" or "float".
        reordering: List of string node names specifying the output order.
        platform: Integer indicating platform of BIE or NEF.

    Returns:
        A list of NumPy arrays of either floats or integers, depending on 'data_type'. If
        data_type is "float", results will be float arrays. If data_type is "fixed", results
        will be fixed arrays.
    """
    try:
        inputs = utils.prep_inputs(pre_results, input_names, True)
        if platform in constants.PLATFORMS_MO3:
            unpack_nefs = unpack_nefs_v1
            get_model_io = get_model_io_v1
        else:
            unpack_nefs = unpack_nefs_v2
            get_model_io = get_model_io_v2

        if nef_file:
            nef = pathlib.Path(nef_file).resolve()
            model_maps, _p_out = unpack_nefs(nef, platform)
            output = csim.csim_inference(nef, inputs, reordering, True, model_maps,
                                         platform=platform, model_id=model_id, data_type=data_type)
        elif onnx_file:
            onnx = pathlib.Path(onnx_file).resolve()
            input_nodes, _, out_node_shape, _ = get_model_io(onnx)
            output = dynasty.dynasty_inference(
                str(onnx), inputs, reordering, True, input_nodes, out_node_shape, data_type=data_type)
        elif bie_file:
            bie = pathlib.Path(bie_file).resolve()
            input_nodes, _, out_node_shape, d_ioinfo = get_model_io(bie, platform)
            output = dynasty.dynasty_inference(
                str(bie), inputs, reordering, True, input_nodes, out_node_shape,
                is_fixed=True, platform=platform, data_type=data_type, d_ioinfo=d_ioinfo)
        else:
            raise exceptions.RequiredConfigError("No input model selected for inference.")
    except exceptions.ConfigError as error:
        sys.exit(error)

    return output

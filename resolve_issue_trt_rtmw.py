import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import warnings

# Numpy compatibility fix for recent versions
warnings.filterwarnings("ignore")
if not hasattr(np, 'bool'): np.bool = bool

LOGGER = trt.Logger(trt.Logger.WARNING)


def init_trt_model(engine_path):
    """
    Initialize TensorRT model for RTMW dynamic batch dual-output inference.
    Returns: (runtime, engine, context, tensor_names)
    """
    runtime = trt.Runtime(LOGGER)
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Get tensor names (update indices if different for your model)
    input_name = engine.get_tensor_name(0)
    output1_name = engine.get_tensor_name(1)
    output2_name = engine.get_tensor_name(2)
    tensor_names = (input_name, output1_name, output2_name)
    return runtime, engine, context, tensor_names


def rtmw_trt_inference(context, tensor_names, input_data):
    """
    Run RTMW TensorRT inference using a safe async stream.
    Args:
        context: TensorRT execution context
        tensor_names: (input_name, output1_name, output2_name)
        input_data: np.ndarray shape (N, 3, 256, 256)  # adjust as needed
    Returns:
        [output1, output2]: list of numpy arrays
    """
    input_name, output1_name, output2_name = tensor_names
    batch_size = input_data.shape[0]
    input_shape = (batch_size, 3, 256, 256)  # adjust if your RTMW input differs

    # set dynamic shape before querying outputs
    context.set_input_shape(input_name, input_shape)
    output1_shape = tuple(context.get_tensor_shape(output1_name))
    output2_shape = tuple(context.get_tensor_shape(output2_name))

    # create CUDA stream
    stream = cuda.Stream()

    # prepare host buffers
    h_input = np.ascontiguousarray(input_data.astype(np.float32).ravel())
    h_output1 = cuda.pagelocked_empty(int(np.prod(output1_shape)), dtype=np.float32)
    h_output2 = cuda.pagelocked_empty(int(np.prod(output2_shape)), dtype=np.float32)

    # allocate device buffers
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output1 = cuda.mem_alloc(h_output1.nbytes)
    d_output2 = cuda.mem_alloc(h_output2.nbytes)

    try:
        # async H->D copy
        cuda.memcpy_htod_async(d_input, h_input, stream)

        # bind tensors
        context.set_tensor_address(input_name, int(d_input))
        context.set_tensor_address(output1_name, int(d_output1))
        context.set_tensor_address(output2_name, int(d_output2))

        # async execution
        context.execute_async_v3(stream_handle=stream.handle)

        # async D->H copy
        cuda.memcpy_dtoh_async(h_output1, d_output1, stream)
        cuda.memcpy_dtoh_async(h_output2, d_output2, stream)

        # wait until all work is done
        stream.synchronize()

        # reshape outputs
        out1 = np.array(h_output1).reshape(output1_shape)
        out2 = np.array(h_output2).reshape(output2_shape)
        return [out1, out2]

    finally:
        # cleanup
        try: stream.synchronize()
        except: pass
        for buf in [d_input, d_output1, d_output2]:
            try: buf.free()
            except: pass

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import warnings

warnings.filterwarnings("ignore")
if not hasattr(np, 'bool'): np.bool = bool

LOGGER = trt.Logger(trt.Logger.WARNING)

def init_trt_model(engine_path):
    """
    Initialize TensorRT model for YOLOX dynamic batch dual-output inference.
    Returns: runtime, engine, context, tensor_names
    Keep `runtime` alive while using engine/context.
    """
    # create runtime and keep it alive (do NOT let it go out of scope while engine used)
    runtime = trt.Runtime(LOGGER)
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    # create context
    context = engine.create_execution_context()

    # Get input/output tensor names (assume input 0, outputs 1/2)
    input_name = engine.get_tensor_name(0)
    output1_name = engine.get_tensor_name(1)
    output2_name = engine.get_tensor_name(2)
    tensor_names = (input_name, output1_name, output2_name)
    return runtime, engine, context, tensor_names


def yolox_trt_inference(context, tensor_names, input_data):
    """
    Run YOLOX TensorRT inference safely using an explicit CUDA stream.
    context: execution context (must be valid)
    tensor_names: (input_name, out1_name, out2_name)
    input_data: np.ndarray shape (N, 3, 640, 640)
    Returns: [output1, output2]
    """
    input_name, output1_name, output2_name = tensor_names
    batch_size = input_data.shape[0]
    input_shape = (batch_size, 3, 640, 640)
    # set the input shape (named tensor API)
    context.set_input_shape(input_name, input_shape)

    # query output shapes AFTER input shape is set (may be dynamic)
    output1_shape = tuple(context.get_tensor_shape(output1_name))
    output2_shape = tuple(context.get_tensor_shape(output2_name))

    # Create a CUDA stream and keep it alive
    stream = cuda.Stream()

    # Use page-locked host memory for stability and performance
    # Flattened sizes
    h_input_flat = np.ascontiguousarray(input_data.astype(np.float32).ravel())
    h_output1_flat = cuda.pagelocked_empty(int(np.prod(output1_shape)), dtype=np.float32)
    h_output2_flat = cuda.pagelocked_empty(int(np.prod(output2_shape)), dtype=np.float32)

    # Allocate device buffers
    d_input = cuda.mem_alloc(h_input_flat.nbytes)
    d_output1 = cuda.mem_alloc(h_output1_flat.nbytes)
    d_output2 = cuda.mem_alloc(h_output2_flat.nbytes)

    try:
        # Async H->D copy
        cuda.memcpy_htod_async(d_input, h_input_flat, stream)

        # Tell TRT where tensors live (device pointers as ints)
        context.set_tensor_address(input_name, int(d_input))
        context.set_tensor_address(output1_name, int(d_output1))
        context.set_tensor_address(output2_name, int(d_output2))

        # Execute asynchronously on the stream (pass actual handle, NOT 0)
        context.execute_async_v3(stream_handle=stream.handle)

        # Async D->H copies
        cuda.memcpy_dtoh_async(h_output1_flat, d_output1, stream)
        cuda.memcpy_dtoh_async(h_output2_flat, d_output2, stream)

        # Synchronize — MUST do this before reading host memory or freeing device buffers
        stream.synchronize()

        # Reshape outputs to original tensor shapes
        h_out1 = np.array(h_output1_flat).reshape(output1_shape)
        h_out2 = np.array(h_output2_flat).reshape(output2_shape)

        return [h_out1, h_out2]

    finally:
        # Safe teardown: make sure stream finished
        try:
            stream.synchronize()
        except Exception:
            pass
        # Free device memory
        try: d_input.free()
        except Exception: pass
        try: d_output1.free()
        except Exception: pass
        try: d_output2.free()
        except Exception: pass
        # let GC handle host pagelocked arrays and stream

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def run_tensorrt_inference(engine_path, input_array):
    """
    Simple TensorRT inference function

    Args:
        engine_path: Path to your .engine file
        input_array: Your numpy array input

    Returns:
        List of output arrays
    """

    # Load the engine
    with open(engine_path, 'rb') as f:
        engine_data = f.read()

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()

    # Get input/output info
    input_shape = engine.get_binding_shape(0)
    output_shape = engine.get_binding_shape(1)
    input_dtype = trt.nptype(engine.get_binding_dtype(0))
    output_dtype = trt.nptype(engine.get_binding_dtype(1))

    # Prepare input
    input_array = input_array.astype(input_dtype)
    if input_array.shape != input_shape:
        input_array = input_array.reshape(input_shape)

    # Allocate memory
    input_size = trt.volume(input_shape)
    output_size = trt.volume(output_shape)

    # Host memory
    h_input = cuda.pagelocked_empty(input_size, input_dtype)
    h_output = cuda.pagelocked_empty(output_size, output_dtype)

    # Device memory
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    # Stream
    stream = cuda.Stream()

    # Copy input to host buffer
    np.copyto(h_input, input_array.ravel())

    # Transfer to GPU
    cuda.memcpy_htod_async(d_input, h_input, stream)

    # Run inference
    context.execute_async_v2([int(d_input), int(d_output)], stream.handle)

    # Transfer result back
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    # Return result
    result = h_output.reshape(output_shape)
    return [result]

# How to use:
if __name__ == "__main__":
    # 1. Replace with your .engine file path
    engine_file = "your_model.engine"

    # 2. Replace with your input data
    # Example for image classification (1, 3, 224, 224):
    my_input = np.random.rand(1, 3, 224, 224).astype(np.float32)

    # 3. Run inference
    try:
        outputs = run_tensorrt_inference(engine_file, my_input)
        print(f"Success! Output shape: {outputs[0].shape}")
        print(f"First 5 values: {outputs[0].flat[:5]}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure:")
        print("1. Your .engine file path is correct")
        print("2. Your input shape matches the model")
        print("3. TensorRT and PyCUDA are installed")

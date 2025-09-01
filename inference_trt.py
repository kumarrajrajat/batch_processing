import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def load_tensorrt_engine(engine_path):
    '''
    Simple function to load a TensorRT engine from file

    Args:
        engine_path (str): Path to .engine file

    Returns:
        tuple: (engine, context, runtime)
    '''
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, 'rb') as f:
        engine_data = f.read()

    engine = runtime.deserialize_cuda_engine(engine_data)
    if engine is None:
        raise RuntimeError(f"Failed to load engine from {engine_path}")

    context = engine.create_execution_context()

    return engine, context, runtime

def allocate_buffers_simple(engine, max_batch_size=1):
    '''
    Simple buffer allocation function

    Args:
        engine: TensorRT engine
        max_batch_size (int): Maximum batch size

    Returns:
        tuple: (inputs, outputs, bindings, stream)
    '''
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    # Handle both new and legacy TensorRT APIs
    if hasattr(engine, 'num_io_tensors'):
        # New API (TensorRT 8.5+)
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(tensor_name)
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
            size = trt.volume(shape) * max_batch_size

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append((host_mem, device_mem, shape, dtype))
            else:
                outputs.append((host_mem, device_mem, shape, dtype))
    else:
        # Legacy API (TensorRT < 8.5)
        for binding in engine:
            shape = engine.get_binding_shape(binding)
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            size = trt.volume(shape) * max_batch_size

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if engine.binding_is_input(binding):
                inputs.append((host_mem, device_mem, shape, dtype))
            else:
                outputs.append((host_mem, device_mem, shape, dtype))

    return inputs, outputs, bindings, stream

def run_inference_simple(context, engine, inputs, outputs, bindings, stream, input_data):
    '''
    Simple inference function

    Args:
        context: TensorRT execution context
        engine: TensorRT engine  
        inputs: Input buffers
        outputs: Output buffers
        bindings: Memory bindings
        stream: CUDA stream
        input_data: Input numpy array

    Returns:
        list: Output numpy arrays
    '''
    # Copy input data to host buffer
    host_input, device_input, input_shape, input_dtype = inputs[0]
    input_data = input_data.astype(input_dtype)

    # Reshape if necessary
    if input_data.shape != input_shape:
        input_data = input_data.reshape(input_shape)

    np.copyto(host_input, input_data.ravel())

    # Transfer input data to device
    cuda.memcpy_htod_async(device_input, host_input, stream)

    # Execute inference
    if hasattr(engine, 'num_io_tensors'):
        # New API
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            context.set_tensor_address(tensor_name, bindings[i])
        context.execute_async_v3(stream_handle=stream.handle)
    else:
        # Legacy API
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Transfer outputs back to host
    result_outputs = []
    for host_output, device_output, output_shape, output_dtype in outputs:
        cuda.memcpy_dtoh_async(host_output, device_output, stream)

    # Wait for all operations to complete
    stream.synchronize()

    # Collect and reshape outputs
    for host_output, device_output, output_shape, output_dtype in outputs:
        output_data = host_output.reshape(output_shape)
        result_outputs.append(output_data)

    return result_outputs

# Complete simple example
def simple_tensorrt_inference(engine_path, input_numpy_array):
    '''
    One-function solution for TensorRT inference

    Args:
        engine_path (str): Path to your .engine file
        input_numpy_array (np.array): Your input data

    Returns:
        list: Model predictions
    '''
    try:
        # Load engine
        engine, context, runtime = load_tensorrt_engine(engine_path)

        # Allocate buffers
        inputs, outputs, bindings, stream = allocate_buffers_simple(engine)

        # Run inference
        results = run_inference_simple(
            context, engine, inputs, outputs, bindings, stream, input_numpy_array
        )

        # Cleanup
        del context, engine, runtime

        return results

    except Exception as e:
        print(f"Error during inference: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Example with dummy data
    engine_path = "your_model.engine"  # Replace with your actual path

    # Create dummy input data - replace this with your actual data
    # Example for an image classification model (batch_size=1, channels=3, height=224, width=224)
    dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)

    print("Running simple TensorRT inference example...")
    print(f"Input shape: {dummy_input.shape}")

    # Run inference
    outputs = simple_tensorrt_inference(engine_path, dummy_input)

    if outputs:
        print(f"Success! Got {len(outputs)} output(s)")
        for i, output in enumerate(outputs):
            print(f"Output {i}: shape={output.shape}, dtype={output.dtype}")
    else:
        print("Inference failed. Please check your engine file path and input data.")

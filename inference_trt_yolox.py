import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import warnings

# Quick fixes for numpy compatibility
warnings.filterwarnings("ignore")
if not hasattr(np, 'bool'): np.bool = bool

def tensorrt_inference_yolox(engine_path, input_batch):
    """
    Run TensorRT inference for YOLOX model with:
    - Input shape: (N, 3, 640, 640)
    - Outputs:
        - output1: (N, 100, 5)
        - output2: (N, 100)
    
    Args:
        engine_path (str): Path to the TensorRT .engine file
        input_batch (np.ndarray): Input numpy array of shape (N, 3, 640, 640)
    
    Returns:
        list: [output1, output2] numpy arrays
    """
    batch_size = input_batch.shape[0]

    # Load TensorRT engine and create execution context
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

    # Retrieve tensor names based on index; adjust if your model differs
    input_name = engine.get_tensor_name(0)
    output1_name = engine.get_tensor_name(1)
    output2_name = engine.get_tensor_name(2)

    # Set input shape dynamically according to batch size
    input_shape = (batch_size, 3, 640, 640)
    context.set_input_shape(input_name, input_shape)

    # Get output shapes after setting input shape
    output1_shape = context.get_tensor_shape(output1_name)
    output2_shape = context.get_tensor_shape(output2_name)

    # Prepare input data with correct dtype and shape
    input_batch = input_batch.astype(np.float32).reshape(input_shape)

    # Allocate host and device memory
    h_input = np.ascontiguousarray(input_batch)
    h_output1 = np.empty(output1_shape, dtype=np.float32)
    h_output2 = np.empty(output2_shape, dtype=np.float32)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output1 = cuda.mem_alloc(h_output1.nbytes)
    d_output2 = cuda.mem_alloc(h_output2.nbytes)

    # Transfer input to the device
    cuda.memcpy_htod(d_input, h_input)

    # Bind tensor addresses and execute inference asynchronously
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output1_name, int(d_output1))
    context.set_tensor_address(output2_name, int(d_output2))
    context.execute_async_v3(stream_handle=0)

    # Transfer outputs back to host
    cuda.memcpy_dtoh(h_output1, d_output1)
    cuda.memcpy_dtoh(h_output2, d_output2)

    # Free CUDA memory
    d_input.free()
    d_output1.free()
    d_output2.free()

    return [h_output1, h_output2]

# Example usage
if __name__ == "__main__":
    engine_file = "your_yolox_model.engine"  # Change to your YOLOX .engine path
    
    # Create random input batch with desirable batch size N (e.g., N=4)
    batch_size = 4
    input_data = np.random.rand(batch_size, 3, 640, 640).astype(np.float32)
    
    outputs = tensorrt_inference_yolox(engine_file, input_data)
    print(f"Output 1 shape: {outputs[0].shape}")  # Expect (N, 100, 5)
    print(f"Output 2 shape: {outputs[1].shape}")  # Expect (N, 100)

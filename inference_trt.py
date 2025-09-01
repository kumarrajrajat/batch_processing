import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import warnings

# Quick fixes
warnings.filterwarnings("ignore")
if not hasattr(np, 'bool'): np.bool = bool

def tensorrt_dynamic_dual_output(engine_path, input_batch):
    """
    Simple TensorRT inference for dynamic batch with dual outputs

    Your model:
    - Input: (-1, 3, 256, 192) -> (batch_size, 3, 256, 192)  
    - Output 1: (-1, 133, 384) -> (batch_size, 133, 384)
    - Output 2: (-1, 133, 512) -> (batch_size, 133, 512)

    Args:
        engine_path: Path to your .engine file
        input_batch: numpy array (batch_size, 3, 256, 192)

    Returns:
        list: [output1, output2] as numpy arrays
    """

    batch_size = input_batch.shape[0]

    # Load engine
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

    # Get tensor names
    input_name = engine.get_tensor_name(0)
    output1_name = engine.get_tensor_name(1)
    output2_name = engine.get_tensor_name(2)

    # Set input shape for current batch size
    input_shape = (batch_size, 3, 256, 192)
    context.set_input_shape(input_name, input_shape)

    # Get resolved output shapes
    output1_shape = context.get_tensor_shape(output1_name)
    output2_shape = context.get_tensor_shape(output2_name)

    # Prepare input
    input_batch = input_batch.astype(np.float32).reshape(input_shape)

    # Allocate memory (regular arrays for stability)
    h_input = np.ascontiguousarray(input_batch)
    h_output1 = np.empty(output1_shape, dtype=np.float32)
    h_output2 = np.empty(output2_shape, dtype=np.float32)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output1 = cuda.mem_alloc(h_output1.nbytes)
    d_output2 = cuda.mem_alloc(h_output2.nbytes)

    # Transfer input to GPU
    cuda.memcpy_htod(d_input, h_input)

    # Set tensor addresses and execute
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output1_name, int(d_output1))
    context.set_tensor_address(output2_name, int(d_output2))

    context.execute_async_v3(stream_handle=0)

    # Get results
    cuda.memcpy_dtoh(h_output1, d_output1)
    cuda.memcpy_dtoh(h_output2, d_output2)

    # Cleanup
    d_input.free()
    d_output1.free()
    d_output2.free()

    return [h_output1, h_output2]

# Simple usage examples
if __name__ == "__main__":
    engine_file = "your_model.engine"  # ← Change this

    # Example 1: Single image (batch_size = 1)
    single_image = np.random.rand(1, 3, 256, 192).astype(np.float32)
    outputs = tensorrt_dynamic_dual_output(engine_file, single_image)
    print(f"Single: Input {single_image.shape} -> Outputs {outputs[0].shape}, {outputs[1].shape}")

    # Example 2: Batch of 20 (your example)
    batch_20 = np.random.rand(20, 3, 256, 192).astype(np.float32)
    outputs = tensorrt_dynamic_dual_output(engine_file, batch_20)
    print(f"Batch 20: Input {batch_20.shape} -> Outputs {outputs[0].shape}, {outputs[1].shape}")

    # Example 3: Any other batch size
    batch_5 = np.random.rand(5, 3, 256, 192).astype(np.float32)
    outputs = tensorrt_dynamic_dual_output(engine_file, batch_5)
    print(f"Batch 5: Input {batch_5.shape} -> Outputs {outputs[0].shape}, {outputs[1].shape}")

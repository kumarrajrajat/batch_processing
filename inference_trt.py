import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import warnings

# FIXES
warnings.filterwarnings("ignore")
if not hasattr(np, 'bool'): np.bool = bool

def fixed_dynamic_tensorrt(engine_path, input_data, batch_size=1):
    """
    Fixed TensorRT inference for dynamic batch models
    Your model: input (-1, 3, 256, 192), output (-1, 133, 384)
    """

    print(f"🔧 Loading dynamic model (batch_size={batch_size})")

    # Load engine
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

    # Use MODERN API (fixes deprecation warning)
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    # Get original shapes (with -1)
    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)

    print(f"📋 Original shapes: input={input_shape}, output={output_shape}")

    # Replace -1 with actual batch_size
    fixed_input_shape = tuple(batch_size if dim == -1 else dim for dim in input_shape)

    print(f"📋 Fixed input shape: {fixed_input_shape}")

    # Set the input shape (this fixes "negative dimensions" error)
    context.set_input_shape(input_name, fixed_input_shape)

    # Get updated output shape
    fixed_output_shape = context.get_tensor_shape(output_name)
    print(f"📋 Fixed output shape: {fixed_output_shape}")

    # Prepare input
    input_data = input_data.astype(np.float32).reshape(fixed_input_shape)

    # Allocate memory (regular arrays - no cuMemHostAlloc error)
    h_input = np.ascontiguousarray(input_data)
    h_output = np.empty(fixed_output_shape, dtype=np.float32)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    # Set tensor addresses (modern API)
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    # Transfer and execute
    cuda.memcpy_htod(d_input, h_input)
    context.execute_async_v3(stream_handle=0)
    cuda.memcpy_dtoh(h_output, d_output)

    # Cleanup
    d_input.free()
    d_output.free()

    print("✅ Dynamic inference completed!")
    return [h_output]

# USAGE for your specific model
if __name__ == "__main__":
    engine_file = "your_model.engine"  # ← Change this
    batch_size = 1  # ← Change this if needed

    # Your model input: (-1, 3, 256, 192) -> (1, 3, 256, 192)
    test_input = np.random.rand(batch_size, 3, 256, 192).astype(np.float32)

    try:
        outputs = fixed_dynamic_tensorrt(engine_file, test_input, batch_size)

        print(f"🎉 SUCCESS!")
        print(f"Output shape: {outputs[0].shape}")  # Should be (1, 133, 384)

    except Exception as e:
        print(f"❌ Error: {e}")

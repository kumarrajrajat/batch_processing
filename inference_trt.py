import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def run_tensorrt_inference(engine_path, input_array):
    """
    Simple TensorRT inference function
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import warnings

# Fix numpy compatibility
warnings.filterwarnings("ignore", message=".*np.bool.*")
if not hasattr(np, 'bool'):
    np.bool = bool

def tensorrt_no_memory_error(engine_path, input_array):
    """
    TensorRT inference that avoids cuMemHostAlloc memory errors
    Uses regular memory instead of pinned memory
    """

    print("🔧 Loading engine without pinned memory...")

    # Load engine
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

    # Get model info
    input_shape = engine.get_binding_shape(0)
    output_shape = engine.get_binding_shape(1)
    input_dtype = trt.nptype(engine.get_binding_dtype(0))
    output_dtype = trt.nptype(engine.get_binding_dtype(1))

    print(f"📋 Input: {input_shape} ({input_dtype.__name__})")
    print(f"📋 Output: {output_shape} ({output_dtype.__name__})")

    # Prepare input
    input_array = input_array.astype(input_dtype).reshape(input_shape)

    # Use REGULAR numpy arrays (NOT pinned memory)
    # This prevents cuMemHostAlloc errors
    host_input = np.ascontiguousarray(input_array)
    host_output = np.empty(output_shape, dtype=output_dtype)

    print(f"💾 Host memory allocated: {host_input.nbytes / (1024*1024):.2f} MB")

    # Allocate GPU memory
    device_input = cuda.mem_alloc(host_input.nbytes)
    device_output = cuda.mem_alloc(host_output.nbytes)

    print("💾 GPU memory allocated successfully")

    # Copy data to GPU
    cuda.memcpy_htod(device_input, host_input)

    # Run inference
    print("⚡ Running inference...")
    success = context.execute_v2([int(device_input), int(device_output)])

    if not success:
        raise RuntimeError("Inference failed")

    # Copy result back
    cuda.memcpy_dtoh(host_output, device_output)

    # Cleanup GPU memory
    device_input.free()
    device_output.free()

    print("✅ Inference completed successfully!")
    return [host_output]

# USAGE EXAMPLE
if __name__ == "__main__":
    print("🚀 TensorRT Memory Error Fix")
    print("=" * 35)

    # Your settings
    engine_file = "your_model.engine"  # ← Change this
    test_input = np.random.rand(1, 3, 224, 224).astype(np.float32)  # ← Change this

    try:
        outputs = tensorrt_no_memory_error(engine_file, test_input)

        print(f"🎉 SUCCESS!")
        for i, output in enumerate(outputs):
            print(f"   Output {i}: {output.shape} {output.dtype}")

    except FileNotFoundError:
        print("❌ Engine file not found - update the engine_file path")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n💡 This version avoids cuMemHostAlloc by using regular memory")
    print("   It's slightly slower but much more reliable!")
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

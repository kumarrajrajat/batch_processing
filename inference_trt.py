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

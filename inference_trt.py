import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import warnings

# Compatibility fixes
warnings.filterwarnings("ignore")
if not hasattr(np, 'bool'): 
    np.bool = bool

class DynamicTensorRTDualOutput:
    """
    TensorRT inference for dynamic batch size model with dual outputs:
    - Input: (-1, 3, 256, 192) - Dynamic batch size
    - Output 1: (-1, 133, 384) - Matches input batch size
    - Output 2: (-1, 133, 512) - Matches input batch size
    """

    def __init__(self, engine_path):
        print("🚀 Initializing Dynamic Batch TensorRT with Dual Outputs")

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        # Load engine
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Analyze model
        self._analyze_dynamic_model()

        print("✅ Dynamic dual output TensorRT ready!")

    def _analyze_dynamic_model(self):
        """Analyze the dynamic model structure"""
        print("\n📋 DYNAMIC MODEL ANALYSIS:")
        print(f"   Total tensors: {self.engine.num_io_tensors}")

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
            tensor_mode = self.engine.get_tensor_mode(tensor_name)

            mode_str = "INPUT" if tensor_mode == trt.TensorIOMode.INPUT else "OUTPUT"
            dynamic_str = " (DYNAMIC)" if -1 in tensor_shape else ""
            print(f"   [{i}] {tensor_name}: {tensor_shape} {tensor_dtype} ({mode_str}){dynamic_str}")

        # Verify expected structure
        expected_input = (-1, 3, 256, 192)
        expected_output1 = (-1, 133, 384)
        expected_output2 = (-1, 133, 512)

        print(f"\n📊 Expected dynamic structure:")
        print(f"   Input: {expected_input}")
        print(f"   Output 1: {expected_output1}")
        print(f"   Output 2: {expected_output2}")

    def infer(self, input_batch, batch_size=None):
        """
        Run inference with any batch size

        Args:
            input_batch: numpy array of shape (batch_size, 3, 256, 192)
            batch_size: Optional, inferred from input if not provided

        Returns:
            list: [output1, output2] where:
                  output1 shape: (batch_size, 133, 384)
                  output2 shape: (batch_size, 133, 512)
        """
        # Infer batch size from input if not provided
        if batch_size is None:
            batch_size = input_batch.shape[0]

        print(f"\n⚡ Running dynamic inference (batch_size = {batch_size})")
        print(f"   Input shape: {input_batch.shape}")

        # Validate input shape
        expected_input_shape = (batch_size, 3, 256, 192)
        if input_batch.shape != expected_input_shape:
            print(f"   ⚠️  Reshaping input: {input_batch.shape} -> {expected_input_shape}")
            input_batch = input_batch.reshape(expected_input_shape)

        input_batch = input_batch.astype(np.float32)

        # Get tensor names
        input_name = self.engine.get_tensor_name(0)
        output1_name = self.engine.get_tensor_name(1) 
        output2_name = self.engine.get_tensor_name(2)

        # Set dynamic input shape
        resolved_input_shape = (batch_size, 3, 256, 192)
        print(f"   Setting input shape: {resolved_input_shape}")
        self.context.set_input_shape(input_name, resolved_input_shape)

        # Get resolved output shapes
        output1_shape = self.context.get_tensor_shape(output1_name)
        output2_shape = self.context.get_tensor_shape(output2_name)

        print(f"   Resolved output 1: {output1_shape}")
        print(f"   Resolved output 2: {output2_shape}")

        # Allocate memory (regular arrays for stability)
        print("💾 Allocating dynamic memory...")

        # Input
        h_input = np.ascontiguousarray(input_batch)
        d_input = cuda.mem_alloc(h_input.nbytes)

        # Output 1
        h_output1 = np.empty(output1_shape, dtype=np.float32)
        d_output1 = cuda.mem_alloc(h_output1.nbytes)

        # Output 2
        h_output2 = np.empty(output2_shape, dtype=np.float32)
        d_output2 = cuda.mem_alloc(h_output2.nbytes)

        total_mb = (h_input.nbytes + h_output1.nbytes + h_output2.nbytes) / (1024 * 1024)
        print(f"   Total memory for batch_{batch_size}: {total_mb:.1f} MB")

        try:
            # Transfer input to GPU
            cuda.memcpy_htod(d_input, h_input)

            # Set tensor addresses
            self.context.set_tensor_address(input_name, int(d_input))
            self.context.set_tensor_address(output1_name, int(d_output1))
            self.context.set_tensor_address(output2_name, int(d_output2))

            # Execute inference
            success = self.context.execute_async_v3(stream_handle=0)
            if not success:
                raise RuntimeError("Dynamic inference execution failed")

            # Transfer outputs back
            cuda.memcpy_dtoh(h_output1, d_output1)
            cuda.memcpy_dtoh(h_output2, d_output2)

            print("   ✅ Dynamic inference completed!")
            print(f"   Output 1: {h_output1.shape}")
            print(f"   Output 2: {h_output2.shape}")

            return [h_output1, h_output2]

        finally:
            # Always cleanup GPU memory
            d_input.free()
            d_output1.free()
            d_output2.free()

    def infer_batch_1(self, single_input):
        """Convenience method for batch_size=1"""
        if len(single_input.shape) == 3:
            # Add batch dimension
            batch_input = single_input[np.newaxis]
        else:
            batch_input = single_input
        return self.infer(batch_input, batch_size=1)

    def infer_batch_any(self, input_list):
        """Inference with any batch size from list of inputs"""
        batch_size = len(input_list)
        batch_input = np.stack(input_list, axis=0)
        return self.infer(batch_input, batch_size=batch_size)

# Simple function for any batch size with dual outputs
def dynamic_dual_output_inference(engine_path, input_batch):
    """
    Simple function for dynamic batch dual output inference

    Args:
        engine_path: Path to .engine file
        input_batch: numpy array of shape (batch_size, 3, 256, 192)

    Returns:
        list: [output1, output2] where:
              output1: (batch_size, 133, 384)
              output2: (batch_size, 133, 512)
    """
    batch_size = input_batch.shape[0]
    print(f"🚀 Dynamic Dual Output (batch_size = {batch_size})")

    # Apply fixes
    if not hasattr(np, 'bool'): 
        np.bool = bool

    # Load engine
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

    # Get tensor information
    input_name = engine.get_tensor_name(0)
    output1_name = engine.get_tensor_name(1)
    output2_name = engine.get_tensor_name(2)

    # Get original shapes (should have -1 for dynamic batch)
    original_input = engine.get_tensor_shape(input_name)
    original_output1 = engine.get_tensor_shape(output1_name)
    original_output2 = engine.get_tensor_shape(output2_name)

    print(f"📋 Original shapes:")
    print(f"   Input: {original_input}")
    print(f"   Output 1: {original_output1}")
    print(f"   Output 2: {original_output2}")

    # Set input shape for current batch size
    resolved_input_shape = (batch_size, 3, 256, 192)
    context.set_input_shape(input_name, resolved_input_shape)

    # Get resolved output shapes
    resolved_output1 = context.get_tensor_shape(output1_name)
    resolved_output2 = context.get_tensor_shape(output2_name)

    print(f"📋 Resolved shapes for batch_{batch_size}:")
    print(f"   Input: {resolved_input_shape}")
    print(f"   Output 1: {resolved_output1}")
    print(f"   Output 2: {resolved_output2}")

    # Validate and prepare input
    input_batch = input_batch.astype(np.float32)
    if input_batch.shape != resolved_input_shape:
        input_batch = input_batch.reshape(resolved_input_shape)

    # Allocate memory
    h_input = np.ascontiguousarray(input_batch)
    h_output1 = np.empty(resolved_output1, dtype=np.float32)
    h_output2 = np.empty(resolved_output2, dtype=np.float32)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output1 = cuda.mem_alloc(h_output1.nbytes)
    d_output2 = cuda.mem_alloc(h_output2.nbytes)

    total_mb = (h_input.nbytes + h_output1.nbytes + h_output2.nbytes) / (1024 * 1024)
    print(f"💾 Memory: {total_mb:.1f} MB")

    try:
        # Transfer and execute
        cuda.memcpy_htod(d_input, h_input)

        context.set_tensor_address(input_name, int(d_input))
        context.set_tensor_address(output1_name, int(d_output1))
        context.set_tensor_address(output2_name, int(d_output2))

        success = context.execute_async_v3(stream_handle=0)
        if not success:
            raise RuntimeError("Inference failed")

        cuda.memcpy_dtoh(h_output1, d_output1)
        cuda.memcpy_dtoh(h_output2, d_output2)

        print(f"✅ Success! Outputs: {h_output1.shape}, {h_output2.shape}")
        return [h_output1, h_output2]

    finally:
        d_input.free()
        d_output1.free()
        d_output2.free()

# Helper functions for different batch sizes
def create_batch_input(data, target_batch_size):
    """
    Create batch input for any target batch size

    Args:
        data: Your input data (various formats)
        target_batch_size: Desired batch size

    Returns:
        numpy array of shape (target_batch_size, 3, 256, 192)
    """
    print(f"📝 Creating batch input for batch_size = {target_batch_size}")

    if isinstance(data, list):
        # List of samples
        available_samples = len(data)
        print(f"   Available samples: {available_samples}")

        if available_samples >= target_batch_size:
            batch = np.stack(data[:target_batch_size], axis=0)
        else:
            # Repeat samples to reach target
            repeated_data = []
            for i in range(target_batch_size):
                repeated_data.append(data[i % available_samples])
            batch = np.stack(repeated_data, axis=0)
            print(f"   Repeated samples to reach {target_batch_size}")

    elif isinstance(data, np.ndarray):
        if len(data.shape) == 3:
            # Single sample - repeat target_batch_size times
            batch = np.tile(data[np.newaxis], (target_batch_size, 1, 1, 1))
            print(f"   Repeated single sample {target_batch_size} times")

        elif len(data.shape) == 4:
            current_batch = data.shape[0]
            if current_batch == target_batch_size:
                batch = data
            elif current_batch > target_batch_size:
                batch = data[:target_batch_size]
                print(f"   Truncated {current_batch} -> {target_batch_size}")
            else:
                # Pad by repeating
                repeats_needed = target_batch_size - current_batch
                padding = np.tile(data[-1:], (repeats_needed, 1, 1, 1))
                batch = np.concatenate([data, padding], axis=0)
                print(f"   Padded {current_batch} -> {target_batch_size}")
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    batch = batch.astype(np.float32)
    expected_shape = (target_batch_size, 3, 256, 192)

    if batch.shape != expected_shape:
        batch = batch.reshape(expected_shape)

    print(f"   ✅ Created: {batch.shape}")
    return batch

# Test function for multiple batch sizes
def test_multiple_batch_sizes(engine_path):
    """Test the model with different batch sizes"""

    print("🧪 TESTING MULTIPLE BATCH SIZES")
    print("=" * 40)

    # Test different batch sizes
    test_batch_sizes = [1, 5, 10, 20, 32]

    for batch_size in test_batch_sizes:
        print(f"\n🔍 Testing batch_size = {batch_size}")
        print("-" * 25)

        try:
            # Create test input for this batch size
            test_input = np.random.rand(batch_size, 3, 256, 192).astype(np.float32)

            # Run inference
            outputs = dynamic_dual_output_inference(engine_path, test_input)

            print(f"   ✅ SUCCESS for batch_{batch_size}")
            print(f"   Input: {test_input.shape}")
            print(f"   Output 1: {outputs[0].shape}")
            print(f"   Output 2: {outputs[1].shape}")

            # Verify shapes match expected pattern
            expected_out1 = (batch_size, 133, 384)
            expected_out2 = (batch_size, 133, 512)

            if outputs[0].shape == expected_out1 and outputs[1].shape == expected_out2:
                print(f"   ✅ Shapes match expected pattern")
            else:
                print(f"   ⚠️  Unexpected shapes!")

        except Exception as e:
            print(f"   ❌ FAILED for batch_{batch_size}: {e}")

if __name__ == "__main__":
    print("🚀 TensorRT Dynamic Batch with Dual Outputs")
    print("=" * 50)

    engine_file = "your_model.engine"  # ← Change this

    try:
        print("\n" + "="*40)
        print("METHOD 1: Using DynamicTensorRTDualOutput class")  
        print("="*40)

        # Method 1: Class-based approach
        dynamic_engine = DynamicTensorRTDualOutput(engine_file)

        # Test with batch_size = 20 (your example)
        test_input_20 = np.random.rand(20, 3, 256, 192).astype(np.float32)
        outputs_20 = dynamic_engine.infer(test_input_20)

        print(f"\n🎉 Batch 20 SUCCESS:")
        print(f"  Output 1: {outputs_20[0].shape}")  # Should be (20, 133, 384)
        print(f"  Output 2: {outputs_20[1].shape}")  # Should be (20, 133, 512)

        # Test with different batch size
        test_input_5 = np.random.rand(5, 3, 256, 192).astype(np.float32)
        outputs_5 = dynamic_engine.infer(test_input_5)

        print(f"\n🎉 Batch 5 SUCCESS:")
        print(f"  Output 1: {outputs_5[0].shape}")   # Should be (5, 133, 384)
        print(f"  Output 2: {outputs_5[1].shape}")   # Should be (5, 133, 512)

        print("\n" + "="*40)
        print("METHOD 2: Testing multiple batch sizes")
        print("="*40)

        # Method 2: Test various batch sizes
        test_multiple_batch_sizes(engine_file)

    except FileNotFoundError:
        print(f"❌ Engine file not found: {engine_file}")
        print("Please update engine_file to your actual .engine file path")

    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "="*50)
    print("✅ DYNAMIC DUAL OUTPUT SUMMARY:")
    print("• Input:  (-1, 3, 256, 192) - Any batch size")
    print("• Output 1: (-1, 133, 384)  - Matches input batch")
    print("• Output 2: (-1, 133, 512)  - Matches input batch")
    print("• Supports: batch_size = 1, 5, 10, 20, 32, etc.")
    print("• Returns: List of 2 numpy arrays")
    print("="*50)

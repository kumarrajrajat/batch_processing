import numpy as np
import tensorrt as trt
import torch
import warnings
import atexit
import gc

# Numpy compatibility fix
warnings.filterwarnings("ignore")
if not hasattr(np, 'bool'): np.bool = bool

# Global cleanup registry
_cleanup_registry = []

def register_cleanup(cleanup_func):
    _cleanup_registry.append(cleanup_func)

def cleanup_all():
    for cleanup_func in reversed(_cleanup_registry):
        try:
            cleanup_func()
        except:
            pass
    _cleanup_registry.clear()

atexit.register(cleanup_all)

def tensorrt_dims_to_tuple(dims):
    """Convert TensorRT Dims to tuple of integers"""
    if hasattr(dims, '__len__'):
        return tuple(dims)
    else:
        # Handle different TensorRT versions
        return tuple([dims[i] for i in range(len(dims))])

class TensorRTInference:
    """TensorRT inference using PyTorch tensors for memory management"""
    
    def __init__(self, engine_path, device='cuda:0'):
        self.engine_path = engine_path
        self.device = torch.device(device)
        self.runtime = None
        self.engine = None
        self.context = None
        self.tensor_names = None
        
        # Set PyTorch CUDA device
        torch.cuda.set_device(self.device)
        
        self._initialize()

    def _initialize(self):
        try:
            with open(self.engine_path, 'rb') as f:
                self.runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
                engine_data = f.read()
                self.engine = self.runtime.deserialize_cuda_engine(engine_data)
            
            if self.engine is None:
                raise RuntimeError("Failed to deserialize engine")
            
            self.context = self.engine.create_execution_context()
            
            # Get tensor names
            input_name = self.engine.get_tensor_name(0)
            output1_name = self.engine.get_tensor_name(1)
            output2_name = self.engine.get_tensor_name(2)
            self.tensor_names = (input_name, output1_name, output2_name)
            
            register_cleanup(self._cleanup)
            print("✅ TensorRT model initialized with PyTorch tensor memory management")
            
        except Exception as e:
            self._cleanup()
            raise e

    def infer(self, input_data):
        """Run inference using PyTorch tensors for memory management"""
        if self.context is None:
            raise RuntimeError("Model not initialized")

        input_name, output1_name, output2_name = self.tensor_names
        
        # Convert input to PyTorch tensor if needed
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data).to(self.device, dtype=torch.float32)
        elif isinstance(input_data, torch.Tensor):
            input_tensor = input_data.to(self.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        batch_size = input_tensor.shape[0]
        
        # Set input shape (adjust as needed for your model)
        input_shape = (batch_size, 3, 256, 256)  # Change for your model
        self.context.set_input_shape(input_name, input_shape)
        
        # Get output shapes and convert TensorRT Dims to tuples
        output1_dims = self.context.get_tensor_shape(output1_name)
        output2_dims = self.context.get_tensor_shape(output2_name)
        
        # Convert TensorRT Dims to Python tuples
        output1_shape = tensorrt_dims_to_tuple(output1_dims)
        output2_shape = tensorrt_dims_to_tuple(output2_dims)
        
        print(f"📋 Tensor shapes:")
        print(f"   Input: {input_shape}")
        print(f"   Output1: {output1_shape} (from {type(output1_dims)})")
        print(f"   Output2: {output2_shape} (from {type(output2_dims)})")
        
        # Ensure input tensor has correct shape
        if input_tensor.shape != input_shape:
            input_tensor = input_tensor.reshape(input_shape)
        
        # Make sure tensor is contiguous
        input_tensor = input_tensor.contiguous()
        
        # Allocate output tensors using PyTorch with converted shapes
        output1_tensor = torch.empty(output1_shape, dtype=torch.float32, device=self.device)
        output2_tensor = torch.empty(output2_shape, dtype=torch.float32, device=self.device)
        
        # Make output tensors contiguous
        output1_tensor = output1_tensor.contiguous()
        output2_tensor = output2_tensor.contiguous()
        
        try:
            # Set tensor addresses using PyTorch tensor data_ptr()
            self.context.set_tensor_address(input_name, input_tensor.data_ptr())
            self.context.set_tensor_address(output1_name, output1_tensor.data_ptr())
            self.context.set_tensor_address(output2_name, output2_tensor.data_ptr())
            
            # Execute inference
            success = self.context.execute_async_v3(stream_handle=0)
            if not success:
                raise RuntimeError("TensorRT inference execution failed")
            
            # Synchronize
            torch.cuda.synchronize()
            
            # Convert outputs back to numpy if needed
            output1_np = output1_tensor.cpu().numpy()
            output2_np = output2_tensor.cpu().numpy()
            
            print(f"✅ Inference completed! Output shapes: {output1_np.shape}, {output2_np.shape}")
            
            return [output1_np, output2_np]
            
        finally:
            # PyTorch handles memory cleanup automatically
            torch.cuda.synchronize()
            gc.collect()

    def infer_torch(self, input_data):
        """Run inference and return PyTorch tensors (no CPU conversion)"""
        if self.context is None:
            raise RuntimeError("Model not initialized")

        input_name, output1_name, output2_name = self.tensor_names
        
        # Convert input to PyTorch tensor if needed
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data).to(self.device, dtype=torch.float32)
        elif isinstance(input_data, torch.Tensor):
            input_tensor = input_data.to(self.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        batch_size = input_tensor.shape[0]
        
        # Set input shape (adjust as needed)
        input_shape = (batch_size, 3, 256, 256)
        self.context.set_input_shape(input_name, input_shape)
        
        # Get output shapes and convert to tuples
        output1_dims = self.context.get_tensor_shape(output1_name)
        output2_dims = self.context.get_tensor_shape(output2_name)
        
        output1_shape = tensorrt_dims_to_tuple(output1_dims)
        output2_shape = tensorrt_dims_to_tuple(output2_dims)
        
        # Ensure input tensor has correct shape and is contiguous
        if input_tensor.shape != input_shape:
            input_tensor = input_tensor.reshape(input_shape)
        input_tensor = input_tensor.contiguous()
        
        # Allocate output tensors with converted shapes
        output1_tensor = torch.empty(output1_shape, dtype=torch.float32, device=self.device).contiguous()
        output2_tensor = torch.empty(output2_shape, dtype=torch.float32, device=self.device).contiguous()
        
        try:
            # Set tensor addresses using data_ptr()
            self.context.set_tensor_address(input_name, input_tensor.data_ptr())
            self.context.set_tensor_address(output1_name, output1_tensor.data_ptr())
            self.context.set_tensor_address(output2_name, output2_tensor.data_ptr())
            
            # Execute inference
            success = self.context.execute_async_v3(stream_handle=0)
            if not success:
                raise RuntimeError("TensorRT inference execution failed")
            
            # Synchronize
            torch.cuda.synchronize()
            
            print(f"✅ Torch inference completed! Output shapes: {output1_tensor.shape}, {output2_tensor.shape}")
            
            # Return PyTorch tensors directly
            return [output1_tensor, output2_tensor]
            
        finally:
            torch.cuda.synchronize()
            gc.collect()

    def _cleanup(self):
        """Safe cleanup of TensorRT resources"""
        try:
            torch.cuda.synchronize()
        except:
            pass
            
        if hasattr(self, 'context') and self.context is not None:
            try:
                del self.context
                self.context = None
            except:
                pass
        
        if hasattr(self, 'engine') and self.engine is not None:
            try:
                del self.engine
                self.engine = None
            except:
                pass
                
        if hasattr(self, 'runtime') and self.runtime is not None:
            try:
                del self.runtime
                self.runtime = None
            except:
                pass
        
        gc.collect()
    
    def __del__(self):
        self._cleanup()

# Convenience functions
def init_trt_model(engine_path, device='cuda:0'):
    return TensorRTInference(engine_path, device)

def rtmw_trt_inference(model, input_data):
    return model.infer(input_data)

def rtmw_trt_inference_torch(model, input_data):
    """Return PyTorch tensors directly"""
    return model.infer_torch(input_data)

# Example usage
if __name__ == "__main__":
    engine_file = "your_rtmw_model.engine"
    
    try:
        # Initialize model
        model = init_trt_model(engine_file)
        
        # Test with numpy input
        print("🧪 Testing with numpy input...")
        input_batch_np = np.random.rand(4, 3, 256, 256).astype(np.float32)
        outputs_np = rtmw_trt_inference(model, input_batch_np)
        
        print(f"✅ Numpy output shapes: {[out.shape for out in outputs_np]}")
        
        # Test with PyTorch tensor input
        print("🧪 Testing with PyTorch tensor input...")
        input_batch_torch = torch.randn(4, 3, 256, 256, device='cuda:0', dtype=torch.float32)
        outputs_torch = rtmw_trt_inference_torch(model, input_batch_torch)
        
        print(f"✅ PyTorch output shapes: {[out.shape for out in outputs_torch]}")
        print(f"✅ Output devices: {[out.device for out in outputs_torch]}")
        
        # Cleanup
        model._cleanup()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        print("🧹 Cleaned up successfully")

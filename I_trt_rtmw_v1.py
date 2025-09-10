import numpy as np
import tensorrt as trt
import cuda  # CUDA Python API (pip install cuda)
import pycuda.autoinit  # Still needed to initialize CUDA context
import warnings
import atexit
import gc

# Numpy compatibility fix
warnings.filterwarnings("ignore")
if not hasattr(np, 'bool'): np.bool = bool

# Global cleanup registry
_cleanup_registry = []

def register_cleanup(cleanup_func):
    """Register cleanup function to be called at exit"""
    _cleanup_registry.append(cleanup_func)

def cleanup_all():
    """Clean up all registered resources"""
    for cleanup_func in reversed(_cleanup_registry):
        try:
            cleanup_func()
        except:
            pass
    _cleanup_registry.clear()

atexit.register(cleanup_all)

class TensorRTInference:
    """Safe TensorRT inference with CUDA Python API memory management"""
    
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.runtime = None
        self.engine = None
        self.context = None
        self.tensor_names = None
        self._initialize()
    
    def _initialize(self):
        """Initialize TensorRT resources"""
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
            
        except Exception as e:
            self._cleanup()
            raise e
    
    def infer(self, input_data):
        """Run inference with safe resource management"""
        if self.context is None:
            raise RuntimeError("Model not initialized")
            
        input_name, output1_name, output2_name = self.tensor_names
        batch_size = input_data.shape[0]
        
        input_shape = (batch_size, 3, 256, 256)  # adjust as needed
        self.context.set_input_shape(input_name, input_shape)
        
        output1_shape = self.context.get_tensor_shape(output1_name)
        output2_shape = self.context.get_tensor_shape(output2_name)
        
        h_input = np.ascontiguousarray(input_data.astype(np.float32).reshape(input_shape))
        h_output1 = np.empty(output1_shape, dtype=np.float32)
        h_output2 = np.empty(output2_shape, dtype=np.float32)
        
        d_input = d_output1 = d_output2 = None
        
        try:
            # Allocate device memory using CUDA Python API
            d_input = cuda.device_array(h_input.nbytes, dtype=np.uint8)
            d_output1 = cuda.device_array(h_output1.nbytes, dtype=np.uint8)
            d_output2 = cuda.device_array(h_output2.nbytes, dtype=np.uint8)
            
            # Copy host to device using cudaMemcpy
            cuda.cudaMemcpy(d_input.device_ctypes_pointer.value, h_input.ctypes.data, h_input.nbytes, cuda.cudaMemcpyHostToDevice)
            
            # Set tensor addresses passing device pointer addresses as integers
            self.context.set_tensor_address(input_name, int(d_input.device_ctypes_pointer.value))
            self.context.set_tensor_address(output1_name, int(d_output1.device_ctypes_pointer.value))
            self.context.set_tensor_address(output2_name, int(d_output2.device_ctypes_pointer.value))
            
            # Execute inference on default stream
            success = self.context.execute_async_v3(stream_handle=0)
            if not success:
                raise RuntimeError("Inference execution failed")
            
            cuda.Context.synchronize()
            
            # Copy device to host memory
            cuda.cudaMemcpy(h_output1.ctypes.data, d_output1.device_ctypes_pointer.value, h_output1.nbytes, cuda.cudaMemcpyDeviceToHost)
            cuda.cudaMemcpy(h_output2.ctypes.data, d_output2.device_ctypes_pointer.value, h_output2.nbytes, cuda.cudaMemcpyDeviceToHost)
            
            cuda.Context.synchronize()
            
            return [h_output1, h_output2]
            
        finally:
            # Free device memory
            if d_output2 is not None:
                try:
                    del d_output2
                except:
                    pass
            if d_output1 is not None:
                try:
                    del d_output1
                except:
                    pass
            if d_input is not None:
                try:
                    del d_input
                except:
                    pass
            
            gc.collect()
    
    def _cleanup(self):
        """Safe resource cleanup"""
        try:
            cuda.Context.synchronize()
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

def init_trt_model(engine_path):
    return TensorRTInference(engine_path)

def rtmw_trt_inference(model, input_data):
    return model.infer(input_data)

# Example usage:
if __name__ == "__main__":
    engine_file = "your_rtmw_model.engine"
    
    try:
        model = init_trt_model(engine_file)
        input_batch = np.random.rand(8, 3, 256, 256).astype(np.float32)
        outputs = rtmw_trt_inference(model, input_batch)
        print(f"Output 1 shape: {outputs[0].shape}")
        print(f"Output 2 shape: {outputs[1].shape}")
        model._cleanup()
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            cuda.Context.synchronize()
            gc.collect()
        except:
            pass
        print("Inference completed - safe exit")

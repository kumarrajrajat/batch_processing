import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
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

# Register cleanup at exit
atexit.register(cleanup_all)

class TensorRTInference:
    """Safe TensorRT inference with proper resource management"""
    
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
            
            # Register cleanup
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
        
        # Set input shape for your model (adjust as needed)
        input_shape = (batch_size, 3, 256, 256)  # RTMW typical shape
        self.context.set_input_shape(input_name, input_shape)
        
        # Get output shapes
        output1_shape = self.context.get_tensor_shape(output1_name)
        output2_shape = self.context.get_tensor_shape(output2_name)
        
        # Prepare input
        h_input = np.ascontiguousarray(input_data.astype(np.float32).reshape(input_shape))
        h_output1 = np.empty(output1_shape, dtype=np.float32)
        h_output2 = np.empty(output2_shape, dtype=np.float32)
        
        # Allocate device memory
        d_input = None
        d_output1 = None 
        d_output2 = None
        
        try:
            d_input = cuda.mem_alloc(h_input.nbytes)
            d_output1 = cuda.mem_alloc(h_output1.nbytes)
            d_output2 = cuda.mem_alloc(h_output2.nbytes)
            
            # Transfer input
            cuda.memcpy_htod(d_input, h_input)
            
            # Set tensor addresses
            self.context.set_tensor_address(input_name, int(d_input))
            self.context.set_tensor_address(output1_name, int(d_output1))
            self.context.set_tensor_address(output2_name, int(d_output2))
            
            # Execute inference (use default stream)
            success = self.context.execute_async_v3(stream_handle=0)
            if not success:
                raise RuntimeError("Inference execution failed")
            
            # Synchronize before memory operations
            cuda.Context.synchronize()
            
            # Transfer outputs
            cuda.memcpy_dtoh(h_output1, d_output1)
            cuda.memcpy_dtoh(h_output2, d_output2)
            
            # Final synchronization
            cuda.Context.synchronize()
            
            return [h_output1, h_output2]
            
        finally:
            # Always free device memory in reverse order
            if d_output2 is not None:
                try:
                    d_output2.free()
                except:
                    pass
            if d_output1 is not None:
                try:
                    d_output1.free()
                except:
                    pass
            if d_input is not None:
                try:
                    d_input.free()
                except:
                    pass
            
            # Force garbage collection
            gc.collect()
    
    def _cleanup(self):
        """Safe cleanup of TensorRT resources"""
        try:
            # Synchronize before cleanup
            cuda.Context.synchronize()
        except:
            pass
            
        # Delete in reverse order of creation
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
        
        # Force garbage collection
        gc.collect()
    
    def __del__(self):
        """Destructor with safe cleanup"""
        self._cleanup()

# Convenience functions matching your original API
def init_trt_model(engine_path):
    """Initialize TensorRT model - returns inference object"""
    return TensorRTInference(engine_path)

def rtmw_trt_inference(model, input_data):
    """Run inference using the model object"""
    return model.infer(input_data)

# Example usage with proper cleanup
if __name__ == "__main__":
    engine_file = "your_rtmw_model.engine"
    
    try:
        # Initialize model
        model = init_trt_model(engine_file)
        
        # Run inference
        input_batch = np.random.rand(8, 3, 256, 256).astype(np.float32)
        outputs = rtmw_trt_inference(model, input_batch)
        
        print(f"Output 1 shape: {outputs[0].shape}")
        print(f"Output 2 shape: {outputs[1].shape}")
        
        # Explicit cleanup (optional - automatic at exit)
        model._cleanup()
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Force final cleanup and synchronization
        try:
            cuda.Context.synchronize()
            gc.collect()
        except:
            pass
        
        print("Inference completed - safe exit")

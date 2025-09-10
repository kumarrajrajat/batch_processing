import numpy as np
import tensorrt as trt
import warnings
import atexit
import gc

# Import low-level CUDA Python APIs
from cuda import Driver, Runtime
import ctypes

# Initialize CUDA Driver
Driver.cuInit(0)

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

def allocate_device_memory(size_bytes):
    dev_ptr = Driver.CUdeviceptr()
    res = Driver.cuMemAlloc_v2(ctypes.byref(dev_ptr), size_bytes)
    if res != 0:
        raise RuntimeError(f'cuMemAlloc_v2 failed with error code {res}')
    return dev_ptr

def copy_host_to_device(host_array, dev_ptr):
    res = Driver.cuMemcpyHtoD_v2(dev_ptr, host_array.ctypes.data, host_array.nbytes)
    if res != 0:
        raise RuntimeError(f'cuMemcpyHtoD_v2 failed with error code {res}')

def copy_device_to_host(dev_ptr, host_array):
    res = Driver.cuMemcpyDtoH_v2(host_array.ctypes.data, dev_ptr, host_array.nbytes)
    if res != 0:
        raise RuntimeError(f'cuMemcpyDtoH_v2 failed with error code {res}')

def free_device_memory(dev_ptr):
    res = Driver.cuMemFree_v2(dev_ptr)
    if res != 0:
        raise RuntimeError(f'cuMemFree_v2 failed with error code {res}')

def synchronize_cuda():
    res = Runtime.cuCtxSynchronize()
    if res != 0:
        raise RuntimeError(f'cuCtxSynchronize failed with error code {res}')

class TensorRTInference:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.runtime = None
        self.engine = None
        self.context = None
        self.tensor_names = None
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

            input_name = self.engine.get_tensor_name(0)
            output1_name = self.engine.get_tensor_name(1)
            output2_name = self.engine.get_tensor_name(2)
            self.tensor_names = (input_name, output1_name, output2_name)

            register_cleanup(self._cleanup)

        except Exception as e:
            self._cleanup()
            raise e

    def infer(self, input_data):
        if self.context is None:
            raise RuntimeError("Model not initialized")

        input_name, output1_name, output2_name = self.tensor_names
        batch_size = input_data.shape[0]

        # Adjust input shape if needed
        input_shape = (batch_size, 3, 256, 256)  # Change according to your model
        self.context.set_input_shape(input_name, input_shape)

        output1_shape = self.context.get_tensor_shape(output1_name)
        output2_shape = self.context.get_tensor_shape(output2_name)

        h_input = np.ascontiguousarray(input_data.astype(np.float32).reshape(input_shape))
        h_output1 = np.empty(output1_shape, dtype=np.float32)
        h_output2 = np.empty(output2_shape, dtype=np.float32)

        d_input = d_output1 = d_output2 = None

        try:
            d_input = allocate_device_memory(h_input.nbytes)
            d_output1 = allocate_device_memory(h_output1.nbytes)
            d_output2 = allocate_device_memory(h_output2.nbytes)

            copy_host_to_device(h_input, d_input)

            self.context.set_tensor_address(input_name, int(d_input))
            self.context.set_tensor_address(output1_name, int(d_output1))
            self.context.set_tensor_address(output2_name, int(d_output2))

            success = self.context.execute_async_v3(stream_handle=0)
            if not success:
                raise RuntimeError("Inference execution failed")

            synchronize_cuda()

            copy_device_to_host(d_output1, h_output1)
            copy_device_to_host(d_output2, h_output2)

            synchronize_cuda()

            return [h_output1, h_output2]

        finally:
            if d_output2 is not None:
                try:
                    free_device_memory(d_output2)
                except:
                    pass
            if d_output1 is not None:
                try:
                    free_device_memory(d_output1)
                except:
                    pass
            if d_input is not None:
                try:
                    free_device_memory(d_input)
                except:
                    pass

            gc.collect()

    def _cleanup(self):
        try:
            synchronize_cuda()
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
            synchronize_cuda()
            gc.collect()
        except:
            pass
        print("Inference completed - safe exit")

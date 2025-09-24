import numpy as np
import torch
import tensorrt as trt
import warnings
import atexit
import gc

warnings.filterwarnings("ignore")
if not hasattr(np, 'bool'): np.bool = bool

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

def torch_dtype_from_trt(dtype: trt.DataType) -> torch.dtype:
    if dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError(f'{dtype} is not supported by torch')

def torch_device_from_trt(device: trt.TensorLocation):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        raise TypeError(f'{device} is not supported by torch')

class TensorRTInference:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.runtime = None
        self.engine = None
        self.context = None
        self._input_names = []
        self._output_names = []
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

            # Load input and output binding names
            names = list(self.engine)
            self._input_names = [n for n in names if self.engine.binding_is_input(n)]
            self._output_names = [n for n in names if not self.engine.binding_is_input(n)]

            register_cleanup(self._cleanup)

        except Exception as e:
            self._cleanup()
            raise e

    def infer(self, inputs: dict):
        """
        Args:
            inputs (dict): mapping input_name -> torch.Tensor (on CUDA device)
        Returns:
            dict: output_name -> torch.Tensor
        """
        if self.context is None:
            raise RuntimeError("Model not initialized")

        bindings = [None] * (len(self._input_names) + len(self._output_names))

        profile_idx = 0

        for input_name, input_tensor in inputs.items():
            idx = self.engine.get_binding_index(input_name)
            profile_shape = self.engine.get_profile_shape(profile_idx, input_name)
            assert input_tensor.dim() == len(profile_shape[0]), \
                f"Input rank mismatch for {input_name}"
            for s_min, s_cur, s_max in zip(profile_shape[0], input_tensor.shape, profile_shape[2]):
                assert s_min <= s_cur <= s_max, \
                    f"Input shape out of profile bounds for {input_name}: {input_tensor.shape}"
            self.context.set_binding_shape(idx, tuple(input_tensor.shape))

            assert input_tensor.is_cuda
            input_tensor = input_tensor.contiguous()

            if input_tensor.dtype == torch.long:
                input_tensor = input_tensor.int()

            bindings[idx] = input_tensor.data_ptr()

        outputs = dict()
        for output_name in self._output_names:
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output_tensor = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output_tensor
            bindings[idx] = output_tensor.data_ptr()

        stream = torch.cuda.current_stream().cuda_stream
        success = self.context.execute_async_v2(bindings, stream_handle=stream)
        if not success:
            raise RuntimeError("Inference execution failed")

        return outputs

    def _cleanup(self):
        try:
            torch.cuda.synchronize()
        except:
            pass
        for attr in ['context', 'engine', 'runtime']:
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    del obj
                    setattr(self, attr, None)
                except:
                    pass
        gc.collect()

    def __del__(self):
        self._cleanup()

# Convenience
def init_trt_model(engine_path):
    return TensorRTInference(engine_path)

def rtmw_trt_inference(model, inputs):
    return model.infer(inputs)

if __name__ == "__main__":
    engine_file = "your_rtmw_model.engine"
    try:
        model = init_trt_model(engine_file)

        input_tensor = torch.randn(8, 3, 256, 256, device='cuda')
        inputs = {'input': input_tensor}  # Use your actual model input name

        outputs = rtmw_trt_inference(model, inputs)
        for name, tensor in outputs.items():
            print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")

        model._cleanup()

    except Exception as e:
        print(f"Error: {e}")

    finally:
        try:
            torch.cuda.synchronize()
            gc.collect()
        except:
            pass
        print("Inference completed - safe exit")

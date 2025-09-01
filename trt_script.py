import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # Initializes CUDA driver

class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)
    return engine

def allocate_buffers(engine, dtype=np.float32):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        shape = engine.get_binding_shape(binding)
        size = trt.volume(shape)
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def infer(engine, context, inputs, outputs, bindings, stream, batch_data):
    np.copyto(inputs.host, batch_data.ravel())
    cuda.memcpy_htod_async(inputs.device, inputs.host, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs.host, outputs.device, stream)
    stream.synchronize()
    output_data = outputs.host
    return output_data

# Usage example
engine_path = "model.engine"  # Path to your .engine file
engine = load_engine(engine_path)
context = engine.create_execution_context()
inputs, outputs, bindings, stream = allocate_buffers(engine, dtype=np.float32)

input_array = np.random.rand(*engine.get_binding_shape(engine)).astype(np.float32)  # Replace with your input
output_array = infer(engine, context, inputs, outputs, bindings, stream, input_array)
output_array = output_array.reshape(engine.get_binding_shape(engine[1]))  # Adjust as needed

print("Output (numpy):", output_array)

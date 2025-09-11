# cuda_stream_manager.py
import torch
import threading

class CUDAStreamManager:
    """Shared CUDA stream manager for cross-module TensorRT operations"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.device = torch.device('cuda:0')
            torch.cuda.set_device(self.device)
            
            # Create separate streams for different operations
            self.tensorrt_stream = torch.cuda.Stream()
            self.pytorch_stream = torch.cuda.Stream() 
            self.default_stream = torch.cuda.default_stream()
            
            self.initialized = True
            print("✅ CUDA Stream Manager initialized")
    
    def get_tensorrt_stream(self):
        """Get stream for TensorRT operations"""
        return self.tensorrt_stream
    
    def get_pytorch_stream(self):
        """Get stream for PyTorch operations"""
        return self.pytorch_stream
    
    def sync_all_streams(self):
        """Synchronize all streams"""
        self.tensorrt_stream.synchronize()
        self.pytorch_stream.synchronize()
        torch.cuda.synchronize()
    
    def cleanup_pytorch_state(self):
        """Clean up PyTorch CUDA state before TensorRT operations"""
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.set_device(self.device)

# Global instance
stream_manager = CUDAStreamManager()

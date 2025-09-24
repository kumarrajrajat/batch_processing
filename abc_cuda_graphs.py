# Detect if CUDA graph already captured
        if not hasattr(self, '_cuda_graph_exec'):
            # Warm-up run before capture
            success = self.context.execute_async_v2(bindings, stream_handle=stream)
            if not success:
                raise RuntimeError("Inference execution failed on warm-up")
            torch_stream.synchronize()

            # Start CUDA Graph capture on this stream
            ctypes.c_void_p(stream)  # ensure stream handle usable for capture in some APIs
            import ctypes
            from ctypes import c_void_p

            # Capture graph
            from cuda import cuda as cuda_drv

            # Start capture (global mode)
            cuda_drv.cudaStreamBeginCapture(c_void_p(stream), cuda_drv.cudaStreamCaptureModeGlobal)

            # Execute inference during capture
            success = self.context.execute_async_v2(bindings, stream_handle=stream)
            if not success:
                raise RuntimeError("Inference execution failed during CUDA Graph capture")

            # End capture
            torch_stream.synchronize()
            graph_handle = c_void_p()
            cuda_drv.cudaStreamEndCapture(c_void_p(stream), ctypes.byref(graph_handle))

            # Instantiate graph exec
            graph_exec_handle = c_void_p()
            cuda_drv.cudaGraphInstantiate(ctypes.byref(graph_exec_handle), graph_handle, None, None, 0)

            # Save graph exec for replay
            self._cuda_graph_exec = graph_exec_handle
            self._cuda_graph = graph_handle

        # Replay captured CUDA Graph
        cuda_drv.cudaGraphLaunch(self._cuda_graph_exec, c_void_p(stream))
        torch_stream.synchronize()

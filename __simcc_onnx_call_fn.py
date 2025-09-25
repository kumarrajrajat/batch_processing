# Convert your existing RTMPose model to end-to-end
create_end_to_end_rtmpose_model(
    "your_rtmpose_model.onnx",      # Your current model
    "rtmpose_end_to_end.onnx",      # New end-to-end model 
    simcc_split_ratio=2.0,          # Your split ratio
    input_size=(256, 192)           # Your input size (W, H)
)



# OLD: Your current workflow with loops + postprocess
# for i in range(0, N, batch_size):
#     outputs = self.model.infer(chuck_imgs)
# keypoints, scores = postprocess([simcc_x, simcc_y], centers, scales)

# NEW: Single inference call 
import onnxruntime as ort

session = ort.InferenceSession("rtmpose_end_to_end.onnx", 
                              providers=['CUDAExecutionProvider'])

final_keypoints, final_scores = session.run(
    ["final_keypoints", "final_scores"],
    {
        "input": batch_images,      # [N, 3, H, W]
        "centers": batch_centers,   # [N, 2] - same as you use now
        "scales": batch_scales      # [N, 2] - same as you use now
    }
)

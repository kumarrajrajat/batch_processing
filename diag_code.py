# DIAGNOSTIC VERSION - Let's find the exact problem
# This version will help identify where the tensor size mismatch occurs

import onnx
from onnx import helper, TensorProto
import numpy as np

def diagnose_rtmpose_model(model_path):
    """
    Diagnostic function to understand the exact tensor shapes in your RTMPose model
    """
    print("🔍 DIAGNOSING RTMPose MODEL SHAPES...")
    
    import onnxruntime as ort
    
    # Load model
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # Print model input/output info
    print("\n📊 MODEL INFO:")
    print("Inputs:")
    for input_info in session.get_inputs():
        print(f"  - {input_info.name}: {input_info.shape} ({input_info.type})")
    
    print("Outputs:")
    for output_info in session.get_outputs():
        print(f"  - {output_info.name}: {output_info.shape} ({output_info.type})")
    
    # Test with small batch to get actual shapes
    print("\n🧪 TESTING WITH SMALL BATCH:")
    test_input = np.random.randn(2, 3, 256, 192).astype(np.float32)
    
    try:
        outputs = session.run(None, {"input": test_input})
        
        print("Actual output shapes:")
        for i, output in enumerate(outputs):
            print(f"  Output {i}: {output.shape} (elements: {output.size})")
            
        # If we have 2 outputs (likely simcc_x and simcc_y)
        if len(outputs) >= 2:
            simcc_x, simcc_y = outputs[0], outputs[1]
            
            print(f"\n📐 DETAILED ANALYSIS:")
            print(f"simcc_x shape: {simcc_x.shape}")
            print(f"simcc_y shape: {simcc_y.shape}")
            
            if len(simcc_x.shape) == 3:  # [N, K, W]
                N, K, W = simcc_x.shape
                H = simcc_y.shape[2] if len(simcc_y.shape) == 3 else simcc_y.shape[1]
                print(f"Detected: N={N}, K={K}, W={W}, H={H}")
                print(f"Total elements after flatten: {N*K} for coords, {N*K} for scores")
                print(f"Target reshape for scores: [{N}, {K}] = {N*K} elements ✓")
                print(f"Target reshape for coords: [{N}, {K}, 2] = {N*K*2} elements")
                
                # This is likely the issue - let's verify
                print(f"\n⚠️  POTENTIAL ISSUE ANALYSIS:")
                print(f"If we have {K} keypoints and batch size {N}:")
                print(f"- After ArgMax: we get {N*K} coordinate values")
                print(f"- After stacking [x,y]: we get {N*K} x 2 = {N*K*2} elements total")
                print(f"- Reshape to [{N}, {K}, 2] needs: {N*K*2} elements ✓")
                print(f"- Reshape to [{N}, {K}] needs: {N*K} elements ✓")
                print("Both should work if we maintain element counts correctly...")
                
        return outputs
        
    except Exception as e:
        print(f"❌ Error running model: {e}")
        return None

def create_minimal_test_version(backbone_path, output_path):
    """
    Create the most minimal possible version to test where the issue occurs
    """
    print("\n🔧 CREATING MINIMAL TEST VERSION...")
    
    onnx_model = onnx.load(backbone_path)
    graph = onnx_model.graph
    
    # Set opset
    del onnx_model.opset_import[:]
    opset = onnx_model.opset_import.add()
    opset.domain = ""
    opset.version = 11
    
    # Add ONLY the most basic processing
    nodes = []
    
    # Just flatten and do ArgMax - no reshaping yet
    simcc_x_flat = helper.make_node('Flatten', ['simcc_x'], ['simcc_x_flat'], axis=1)
    nodes.append(simcc_x_flat)
    
    simcc_y_flat = helper.make_node('Flatten', ['simcc_y'], ['simcc_y_flat'], axis=1)  
    nodes.append(simcc_y_flat)
    
    # Get shapes for debugging
    shape_x_flat = helper.make_node('Shape', ['simcc_x_flat'], ['shape_x_flat'])
    nodes.append(shape_x_flat)
    
    # Just pass through the flattened tensors as outputs for debugging
    output_x = helper.make_node('Identity', ['simcc_x_flat'], ['debug_x_flat'])
    nodes.append(output_x)
    
    output_y = helper.make_node('Identity', ['simcc_y_flat'], ['debug_y_flat'])
    nodes.append(output_y)
    
    output_shape = helper.make_node('Identity', ['shape_x_flat'], ['debug_shape'])
    nodes.append(output_shape)
    
    # Add nodes to graph
    graph.node.extend(nodes)
    
    # Update outputs to debug values
    del graph.output[:]
    debug_x_output = helper.make_tensor_value_info('debug_x_flat', TensorProto.FLOAT, [None, None])
    debug_y_output = helper.make_tensor_value_info('debug_y_flat', TensorProto.FLOAT, [None, None])
    debug_shape_output = helper.make_tensor_value_info('debug_shape', TensorProto.INT64, [None])
    
    graph.output.extend([debug_x_output, debug_y_output, debug_shape_output])
    
    # Save
    onnx.save(onnx_model, output_path)
    print(f"✅ Minimal test version saved: {output_path}")
    
    return output_path

def test_minimal_version(model_path):
    """Test the minimal version to see actual tensor shapes"""
    print(f"\n🧪 TESTING MINIMAL VERSION: {model_path}")
    
    import onnxruntime as ort
    
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # Test with small batch
    test_input = np.random.randn(2, 3, 256, 192).astype(np.float32)
    
    try:
        outputs = session.run(None, {"input": test_input})
        
        debug_x, debug_y, debug_shape = outputs
        
        print(f"📊 MINIMAL VERSION RESULTS:")
        print(f"Flattened simcc_x shape: {debug_x.shape}")
        print(f"Flattened simcc_y shape: {debug_y.shape}")  
        print(f"Shape tensor value: {debug_shape}")
        print(f"Total elements in debug_x: {debug_x.size}")
        print(f"Total elements in debug_y: {debug_y.size}")
        
        # Calculate what the original shape should be
        if len(debug_shape) >= 2:
            flat_batch_size = debug_shape[0]  # Should be N*K
            flat_width = debug_shape[1]       # Should be W or H
            print(f"Flattened: [{flat_batch_size}, {flat_width}]")
            
            # Try to deduce original shape
            if flat_batch_size % 2 == 0:  # If batch was 2
                likely_K = flat_batch_size // 2
                print(f"Likely original shape: [2, {likely_K}, {flat_width}]")
                print(f"So probably N=2, K={likely_K}")
                
                # This tells us what the reshape should be
                print(f"\n💡 FOR RESHAPE:")
                print(f"After ArgMax, we'll have {flat_batch_size} values")
                print(f"To reshape to [2, {likely_K}]: need {2 * likely_K} = {flat_batch_size} elements ✓")
                print(f"To reshape to [2, {likely_K}, 2]: need {2 * likely_K * 2} = {flat_batch_size * 2} elements")
                
        return outputs
        
    except Exception as e:
        print(f"❌ Error testing minimal version: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 RTMPose DIAGNOSTIC TOOL")
    print("This will help identify the exact cause of the reshape error\n")
    
    # Step 1: Diagnose the original model
    backbone_model = "rtmpose_backbone.onnx"  # Replace with your model path
    
    try:
        print("=" * 60)
        print("STEP 1: ANALYZING ORIGINAL MODEL")
        print("=" * 60)
        
        original_outputs = diagnose_rtmpose_model(backbone_model)
        
        if original_outputs is not None:
            print("=" * 60)
            print("STEP 2: CREATING MINIMAL TEST VERSION")  
            print("=" * 60)
            
            minimal_model = create_minimal_test_version(backbone_model, "rtmpose_minimal_test.onnx")
            
            print("=" * 60)
            print("STEP 3: TESTING MINIMAL VERSION")
            print("=" * 60)
            
            minimal_outputs = test_minimal_version(minimal_model)
            
            print("=" * 60)
            print("ANALYSIS COMPLETE")
            print("=" * 60)
            
            print("\n📋 SUMMARY:")
            print("Please share the output above so we can identify:")
            print("1. The exact tensor shapes at each step")
            print("2. Where the element count mismatch occurs") 
            print("3. The correct reshape dimensions to use")
            print("\nThis will help create a working end-to-end model!")
    
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 ALTERNATIVE APPROACH:")
        print("If the diagnostic fails, please manually check:")
        print("1. What are your model's actual input/output names?")
        print("2. What shapes do you get when you run inference?")
        print("3. How many keypoints does your model have?")
        print("4. Share the exact error message you're getting")

"""
🎯 DIAGNOSTIC APPROACH:

Run this diagnostic script to get the exact tensor shapes and identify where the mismatch occurs.

The error suggests we have 4 elements trying to reshape to [4, 133] (532 elements).
This diagnostic will show us:

1. The actual model input/output shapes
2. The tensor sizes after each operation
3. Where elements are being lost or miscounted
4. The correct reshape dimensions to use

Please run this and share the output - it will help us fix the reshape issue!
"""

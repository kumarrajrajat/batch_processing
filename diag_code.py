# ULTRA MINIMAL VERSION - Just Reshape Test
# Let's isolate the exact reshape that's failing

import onnx
from onnx import helper, TensorProto
import numpy as np

def create_minimal_reshape_test(backbone_path, output_path):
    """
    Create the most minimal possible version to test ONLY the reshape that's failing
    Based on your diagnostic: the issue is likely in a specific reshape operation
    """
    
    print("🔧 Creating minimal reshape test...")
    
    onnx_model = onnx.load(backbone_path)
    graph = onnx_model.graph
    
    # Clear everything and start fresh
    del onnx_model.opset_import[:]
    opset = onnx_model.opset_import.add()
    opset.domain = ""
    opset.version = 11
    
    # Clear existing nodes - we'll add minimal ones
    del graph.node[:]
    
    nodes = []
    constants = []
    
    # === STEP 1: Basic operations only (no complex logic) ===
    
    # Flatten inputs (this should work fine)
    simcc_x_flat = helper.make_node('Flatten', ['simcc_x'], ['simcc_x_flat'], axis=1)
    nodes.append(simcc_x_flat)
    
    simcc_y_flat = helper.make_node('Flatten', ['simcc_y'], ['simcc_y_flat'], axis=1)
    nodes.append(simcc_y_flat)
    
    # ArgMax (this should work fine)  
    x_locs = helper.make_node('ArgMax', ['simcc_x_flat'], ['x_locs'], axis=1, keepdims=0)
    nodes.append(x_locs)
    
    y_locs = helper.make_node('ArgMax', ['simcc_y_flat'], ['y_locs'], axis=1, keepdims=0)
    nodes.append(y_locs)
    
    # Get scores (this should work fine)
    max_x = helper.make_node('ReduceMax', ['simcc_x_flat'], ['max_x'], axes=[1], keepdims=0)
    nodes.append(max_x)
    
    max_y = helper.make_node('ReduceMax', ['simcc_y_flat'], ['max_y'], axes=[1], keepdims=0)
    nodes.append(max_y)
    
    scores_sum = helper.make_node('Add', ['max_x', 'max_y'], ['scores_sum'])
    nodes.append(scores_sum)
    
    scores_flat = helper.make_node('Mul', ['scores_sum', 'half'], ['scores_flat'])
    nodes.append(scores_flat)
    
    # Convert coords to float and stack
    x_locs_float = helper.make_node('Cast', ['x_locs'], ['x_locs_float'], to=TensorProto.FLOAT)
    nodes.append(x_locs_float)
    
    y_locs_float = helper.make_node('Cast', ['y_locs'], ['y_locs_float'], to=TensorProto.FLOAT)
    nodes.append(y_locs_float)
    
    x_unsqueeze = helper.make_node('Unsqueeze', ['x_locs_float'], ['x_unsqueeze'], axes=[-1])
    nodes.append(x_unsqueeze)
    
    y_unsqueeze = helper.make_node('Unsqueeze', ['y_locs_float'], ['y_unsqueeze'], axes=[-1])
    nodes.append(y_unsqueeze)
    
    coords_flat = helper.make_node('Concat', ['x_unsqueeze', 'y_unsqueeze'], ['coords_flat'], axis=-1)
    nodes.append(coords_flat)
    
    # === STEP 2: The minimal reshape test ===
    
    # Get shape info
    shape_x = helper.make_node('Shape', ['simcc_x'], ['shape_x'])
    nodes.append(shape_x)
    
    # Extract dimensions
    n_dim = helper.make_node('Gather', ['shape_x', 'zero_idx'], ['N'], axis=0)
    nodes.append(n_dim)
    
    k_dim = helper.make_node('Gather', ['shape_x', 'one_idx'], ['K'], axis=0)
    nodes.append(k_dim)
    
    # Convert to 1D tensors  
    n_1d = helper.make_node('Unsqueeze', ['N'], ['N_1d'], axes=[0])
    nodes.append(n_1d)
    
    k_1d = helper.make_node('Unsqueeze', ['K'], ['K_1d'], axes=[0])
    nodes.append(k_1d)
    
    # === CRITICAL TEST: Build shapes step by step ===
    
    # First, try to build [N, K] shape (simpler)
    nk_shape = helper.make_node('Concat', ['N_1d', 'K_1d'], ['nk_shape'], axis=0)
    nodes.append(nk_shape)
    
    # Test 1: Can we reshape scores? scores_flat is [N*K] -> [N, K]  
    scores_reshaped = helper.make_node('Reshape', ['scores_flat', 'nk_shape'], ['scores_reshaped'])
    nodes.append(scores_reshaped)
    
    # Build [N, K, 2] shape
    nk2_shape = helper.make_node('Concat', ['N_1d', 'K_1d', 'two_1d'], ['nk2_shape'], axis=0)
    nodes.append(nk2_shape)
    
    # Test 2: Can we reshape coords? coords_flat is [N*K, 2] -> [N, K, 2]
    coords_reshaped = helper.make_node('Reshape', ['coords_flat', 'nk2_shape'], ['coords_reshaped'])
    nodes.append(coords_reshaped)
    
    # === OUTPUT: Just pass through the reshaped tensors ===
    final_coords = helper.make_node('Identity', ['coords_reshaped'], ['final_keypoints'])
    nodes.append(final_coords)
    
    final_scores = helper.make_node('Identity', ['scores_reshaped'], ['final_scores'])
    nodes.append(final_scores)
    
    # Add constants
    constants.append(helper.make_tensor('zero_idx', TensorProto.INT64, [], [0]))
    constants.append(helper.make_tensor('one_idx', TensorProto.INT64, [], [1]))
    constants.append(helper.make_tensor('two_1d', TensorProto.INT64, [1], [2]))
    constants.append(helper.make_tensor('half', TensorProto.FLOAT, [], [0.5]))
    
    # Add everything to graph
    graph.node.extend(nodes)
    graph.initializer.extend(constants)
    
    # Update outputs
    del graph.output[:]
    keypoints_output = helper.make_tensor_value_info('final_keypoints', TensorProto.FLOAT, [None, None, 2])
    scores_output = helper.make_tensor_value_info('final_scores', TensorProto.FLOAT, [None, None])
    graph.output.extend([keypoints_output, scores_output])
    
    # Save
    onnx.save(onnx_model, output_path)
    print(f"✅ Minimal reshape test saved: {output_path}")
    
    return output_path

def test_minimal_reshape(model_path, batch_size=2):
    """Test the minimal reshape version"""
    
    import onnxruntime as ort
    
    print(f"\n🧪 Testing minimal reshape: {model_path}")
    
    # Use small batch
    test_input = np.random.randn(batch_size, 3, 256, 192).astype(np.float32)
    
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        outputs = session.run(None, {"input": test_input})
        
        final_keypoints, final_scores = outputs
        
        print("✅ MINIMAL RESHAPE TEST PASSED!")
        print(f"   Keypoints: {final_keypoints.shape}")
        print(f"   Scores: {final_scores.shape}")
        print(f"   Expected: keypoints({batch_size}, 133, 2), scores({batch_size}, 133)")
        
        if final_keypoints.shape == (batch_size, 133, 2) and final_scores.shape == (batch_size, 133):
            print("🎉 RESHAPE WORKS! The issue is in the complex masking logic.")
            print("   Solution: Simplify the masking operations.")
        else:
            print("❌ Basic reshape failed. Need to debug tensor shapes.")
            
        return outputs
        
    except Exception as e:
        print(f"❌ MINIMAL RESHAPE FAILED: {e}")
        
        # This tells us exactly what's wrong
        if "reshape" in str(e).lower():
            print("\n💡 The basic reshape is failing. This means:")
            print("   1. The tensor element counts don't match")
            print("   2. There might be an issue with shape extraction")
            print("   3. The ArgMax or stacking operations are wrong")
            
        import traceback
        traceback.print_exc()
        return None

def create_debug_version(backbone_path, output_path):
    """Create version that outputs intermediate tensor shapes for debugging"""
    
    print("🔍 Creating debug version to see actual tensor shapes...")
    
    onnx_model = onnx.load(backbone_path)
    graph = onnx_model.graph
    
    del onnx_model.opset_import[:]
    opset = onnx_model.opset_import.add()
    opset.domain = ""
    opset.version = 11
    
    del graph.node[:]
    nodes = []
    constants = []
    
    # Basic operations
    simcc_x_flat = helper.make_node('Flatten', ['simcc_x'], ['simcc_x_flat'], axis=1)
    nodes.append(simcc_x_flat)
    
    simcc_y_flat = helper.make_node('Flatten', ['simcc_y'], ['simcc_y_flat'], axis=1)
    nodes.append(simcc_y_flat)
    
    # Get shapes of flattened tensors
    shape_x_flat = helper.make_node('Shape', ['simcc_x_flat'], ['shape_x_flat'])
    nodes.append(shape_x_flat)
    
    shape_y_flat = helper.make_node('Shape', ['simcc_y_flat'], ['shape_y_flat'])
    nodes.append(shape_y_flat)
    
    # ArgMax
    x_locs = helper.make_node('ArgMax', ['simcc_x_flat'], ['x_locs'], axis=1, keepdims=0)
    nodes.append(x_locs)
    
    y_locs = helper.make_node('ArgMax', ['simcc_y_flat'], ['y_locs'], axis=1, keepdims=0)
    nodes.append(y_locs)
    
    # Get shapes after ArgMax
    shape_x_locs = helper.make_node('Shape', ['x_locs'], ['shape_x_locs'])
    nodes.append(shape_x_locs)
    
    shape_y_locs = helper.make_node('Shape', ['y_locs'], ['shape_y_locs'])
    nodes.append(shape_y_locs)
    
    # Get original shape
    shape_x = helper.make_node('Shape', ['simcc_x'], ['shape_x'])
    nodes.append(shape_x)
    
    # Extract N and K
    n_dim = helper.make_node('Gather', ['shape_x', 'zero_idx'], ['N'], axis=0)
    nodes.append(n_dim)
    
    k_dim = helper.make_node('Gather', ['shape_x', 'one_idx'], ['K'], axis=0)
    nodes.append(k_dim)
    
    # Output all the shapes for debugging
    debug1 = helper.make_node('Identity', ['shape_x_flat'], ['debug_shape_x_flat'])
    nodes.append(debug1)
    
    debug2 = helper.make_node('Identity', ['shape_x_locs'], ['debug_shape_x_locs']) 
    nodes.append(debug2)
    
    debug3 = helper.make_node('Identity', ['shape_x'], ['debug_shape_x'])
    nodes.append(debug3)
    
    debug4 = helper.make_node('Identity', ['N'], ['debug_N'])
    nodes.append(debug4)
    
    debug5 = helper.make_node('Identity', ['K'], ['debug_K'])
    nodes.append(debug5)
    
    constants.append(helper.make_tensor('zero_idx', TensorProto.INT64, [], [0]))
    constants.append(helper.make_tensor('one_idx', TensorProto.INT64, [], [1]))
    
    graph.node.extend(nodes)
    graph.initializer.extend(constants)
    
    # Update outputs to debug shapes
    del graph.output[:]
    debug_outputs = [
        helper.make_tensor_value_info('debug_shape_x_flat', TensorProto.INT64, [None]),
        helper.make_tensor_value_info('debug_shape_x_locs', TensorProto.INT64, [None]),
        helper.make_tensor_value_info('debug_shape_x', TensorProto.INT64, [None]),
        helper.make_tensor_value_info('debug_N', TensorProto.INT64, []),
        helper.make_tensor_value_info('debug_K', TensorProto.INT64, [])
    ]
    graph.output.extend(debug_outputs)
    
    onnx.save(onnx_model, output_path)
    print(f"✅ Debug version saved: {output_path}")
    return output_path

def test_debug_version(model_path, batch_size=2):
    """Test debug version to see actual shapes"""
    
    import onnxruntime as ort
    
    print(f"\n🔍 Testing debug version: {model_path}")
    
    test_input = np.random.randn(batch_size, 3, 256, 192).astype(np.float32)
    
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        outputs = session.run(None, {"input": test_input})
        
        shape_x_flat, shape_x_locs, shape_x, N, K = outputs
        
        print("🔍 DEBUG RESULTS:")
        print(f"   Original simcc_x shape: {shape_x}")
        print(f"   After flatten shape: {shape_x_flat}")
        print(f"   After ArgMax shape: {shape_x_locs}")
        print(f"   Extracted N: {N}")
        print(f"   Extracted K: {K}")
        
        print(f"\n📊 ELEMENT COUNT ANALYSIS:")
        flat_elements = np.prod(shape_x_flat)
        argmax_elements = np.prod(shape_x_locs)
        expected_nk = N * K
        
        print(f"   Flattened tensor elements: {flat_elements}")
        print(f"   ArgMax result elements: {argmax_elements}")  
        print(f"   Expected N*K: {expected_nk}")
        print(f"   Match: {argmax_elements == expected_nk}")
        
        if argmax_elements != expected_nk:
            print("❌ FOUND THE ISSUE: ArgMax element count doesn't match N*K!")
            print("   This explains the reshape failure.")
        else:
            print("✅ Element counts look correct. Issue must be elsewhere.")
            
        return outputs
        
    except Exception as e:
        print(f"❌ Debug version failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 MINIMAL RESHAPE DIAGNOSTIC")
    print("This will isolate the exact reshape operation that's failing\n")
    
    backbone_model = "rtmpose_backbone.onnx"  # Replace with your model path
    
    try:
        print("=" * 60)
        print("STEP 1: DEBUG VERSION - Check tensor shapes")
        print("=" * 60)
        
        debug_model = create_debug_version(backbone_model, "debug_shapes.onnx")
        debug_results = test_debug_version(debug_model, batch_size=2)
        
        if debug_results is not None:
            print("=" * 60)
            print("STEP 2: MINIMAL RESHAPE TEST")
            print("=" * 60)
            
            minimal_model = create_minimal_reshape_test(backbone_model, "minimal_reshape_test.onnx")
            minimal_results = test_minimal_reshape(minimal_model, batch_size=2)
            
            print("=" * 60)
            print("ANALYSIS COMPLETE")
            print("=" * 60)
            
            if minimal_results is not None:
                print("✅ BASIC RESHAPE WORKS!")
                print("   The issue is in the complex masking/processing logic.")
                print("   Solution: We need to simplify the masking operations.")
            else:
                print("❌ BASIC RESHAPE FAILS!")
                print("   The issue is fundamental - in ArgMax or shape extraction.")
                print("   Solution: Need to fix the basic tensor operations.")
                
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n💡 NEXT STEPS:")
    print("1. Run this diagnostic")
    print("2. Share the output showing exactly where it fails")
    print("3. I'll create a targeted fix for the specific issue")

"""
🎯 THIS WILL TELL US EXACTLY WHAT'S WRONG:

This diagnostic will:
1. Check if basic ArgMax and shape extraction work
2. Test if simple reshape operations work  
3. Isolate whether the issue is in basic operations or complex masking
4. Show exact tensor shapes at each step

Please run this and share the output - it will pinpoint the exact problem!
"""

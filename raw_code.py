frame_index = 0 
for frame_index in tqdm(range(total_frmaes)):
    if frame_index < start_frme:
        continue
    if end_frame > 0 and frame_index >= end_frame:
        break
    
    success, frame = cap.read()
    if not success:
        break
    input_img , ratio = resize_with_aspect_ratio(frame, det_input_size[0], det_input_size[1], pad_value = 114)
    keypoints, bboxes, scores = wholebody(input_img)
    
    keypoints[..., :2] *= ratio
    bboxes[..., :4] *= ratio
    
    data_per_frame = []
    remain_kps = []
    remain_bboxes = []
    
    for bbox, kps, in zip(bboxes, keypoints):
        kps_list, bbox_list = [],[]
        data_per_frame.append({"bbox":bbox.tolist(),
            "kps":kps.tolist()
        })
        remain_kps.append(kps)
        remain_bboxes.append(bbox)
        
    out_data[frame_index] = data_per_frame
    idx += 1
    
    img = frame.copy()
    
    if len(remain_kps):
        img = draw_bbox(img, remain_bboxes)
        img = draw_skeleton(img,
        np.array(remain_kps),
        scores,
        kpt_thr = 0.1,
        line_width = 2)
        
    if save_plot:
        out.write(cv2.resize(image,out_size))
        

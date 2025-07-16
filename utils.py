import os
import cv2
import numpy as np
import json
import torch
from trt_inf import TRTInference
# from pycocotools.coco import COCO
# from pycocotools.cocoeval_modified import COCOeval
from pycocotools_mod.coco import COCO
from pycocotools_mod.cocoeval_modified import COCOeval

# Try to import TensorRT for engine building
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    print("Warning: TensorRT not available. Engine building will be disabled.")
    TRT_AVAILABLE = False


def f1_score(predictions_path, ground_truths_path):
    coco_gt = COCO(ground_truths_path)

    gt_image_ids = coco_gt.getImgIds()

    with open(predictions_path, 'r') as f:
        detection_data = json.load(f)
    filtered_detection_data = [
        item for item in detection_data if item['image_id'] in gt_image_ids]
    with open('./temp.json', 'w') as f:
        json.dump(filtered_detection_data, f)
    coco_dt = coco_gt.loadRes('./temp.json')
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Assuming the F1 score is at index 20 in the stats array
    return coco_eval.stats[20]  # Return the F1 score from the evaluation stats
    # return 0.85  # Simulated constant value for demo purposes


def changeId(id):
    sceneList = ['M', 'A', 'E', 'N']
    cameraId = int(id.split('_')[0].split('camera')[1])
    sceneId = sceneList.index(id.split('_')[1])
    frameId = int(id.split('_')[2])
    imageId = int(str(cameraId)+str(sceneId)+str(frameId))
    return imageId

def is_grayscale(input_image, thres=1e-5, return_max_diff=False):
    """Check if image is grayscale (night) or color (day)"""
    if isinstance(input_image, str):
        img_bgr = cv2.imread(input_image, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Unable to read image: {input_image}")
    else:
        img_bgr = input_image.copy()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    arr = img_rgb.astype(np.float32) / 255.0
    h, w, _ = arr.shape
    h1 = h // 4
    h2 = h - h1
    w1 = w // 4
    w2 = w - w1
    diff_rg = np.max(np.abs(arr[h1:h2, w1:w2, 0] - arr[h1:h2, w1:w2, 1]))
    # diff_rb = np.max(np.abs(arr[h1:h2, w1:w2, 0] - arr[h1:h2, w1:w2, 2]))
    diff_gb = np.max(np.abs(arr[h1:h2, w1:w2, 1] - arr[h1:h2, w1:w2, 2]))
    max_diff = max(diff_rg, diff_gb)
    # print(max_diff)
    if return_max_diff:
        return max_diff <= thres, max_diff
    return max_diff <= thres

def get_model(model_path):
    """Initialize model - can be TensorRT engine or ONNX file (will build engine automatically)"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Check if it's an ONNX file that needs to be converted to TensorRT engine
    if model_path.endswith('.onnx'):
        # Generate engine path based on ONNX path
        engine_path = model_path.replace('.onnx', '.engine')
        
        # Check if engine already exists and is newer than ONNX
        if not os.path.exists(engine_path):
            print(f"Building TensorRT engine from ONNX: {model_path} -> {engine_path}")
            build_engine_from_onnx(model_path, engine_path, 1610612736)
        
        # Use the engine file
        model_path = engine_path
    
    # Check if it's a TensorRT engine file
    if model_path.endswith('.engine'):
        # Create a wrapper class to make TensorRT model compatible with YOLO-like interface
        class TRTModelWrapper:
            def __init__(self, engine_path, device):
                self.model = TRTInference(engine_path, device=device, backend="torch")
                self.device = device
            
            def __call__(self, img, verbose=False):
                # Convert the preprocessed image back to blob format for TensorRT
                if isinstance(img, np.ndarray):
                    # Ensure image is in correct format (BGR -> RGB)
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        # Convert BGR to RGB
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else:
                        img_rgb = img
                    
                    # Get original dimensions
                    h, w = img_rgb.shape[:2]
                    orig_size = torch.tensor([w, h])[None].to(self.device)
                    
                    # Resize to model input size (1280x1280)
                    img_resized = cv2.resize(img_rgb, (1280, 1280))
                    
                    # Convert to tensor format: HWC -> CHW and normalize to [0,1]
                    img_tensor = torch.from_numpy(img_resized).float() / 255.0
                    img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
                    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
                    
                    blob = {
                        "images": img_tensor.to(self.device),
                        "orig_target_sizes": orig_size.to(self.device),
                    }
                    
                    return self.model(blob)
                else:
                    return self.model(img)
        
        return TRTModelWrapper(model_path, device)
    else:
        # For other model formats (like YOLO .pt files), would need YOLO import
        raise NotImplementedError(f"Model format not supported for {model_path}. Use .onnx or .engine files.")


def preprocess_image(img, device="cuda:0"):
    """Preprocess image for model input
    Args:
        img: CV2 image (numpy array)
        device: Device to use (optional, for compatibility)
    Returns:
        Preprocessed image (for original interface compatibility)
    """
    if img is None:
        raise ValueError("Input image is None.")
    
    # For the original interface, we need to return the image for model(img, verbose=False)
    # Since TensorRT expects a blob, we'll return the CV2 image and let the model wrapper handle it
    return img


def postprocess_result(output, confidence_threshold=0.65):
    """Convert model output to required format [boxes, scores, classes]
    Args:
        output: Model output (can be TensorRT output dict or other model results)
        confidence_threshold: Confidence threshold for filtering detections
    Returns:
        List of [boxes, scores, classes] in the format expected by run_evaluation_jetson.py
    """
    # print(f"Confidence Threshold: {confidence_threshold}")
    try:
        # Handle TensorRT model output (dictionary format)
        if isinstance(output, dict) and "scores" in output:
            # Get raw outputs
            raw_scores = output["scores"]
            raw_boxes = output["boxes"]
            raw_labels = output["labels"]
            
            # Create a combined mask for valid detections (no NaN in any field)
            valid_score_mask = ~torch.isnan(raw_scores)
            valid_box_mask = ~torch.isnan(raw_boxes).any(dim=-1)
            valid_label_mask = ~torch.isnan(raw_labels)
            
            # Combine all validity masks
            valid_mask = valid_score_mask & valid_box_mask & valid_label_mask
            
            if valid_mask.sum() == 0:
                return [[], [], []]
            
            # Apply validity mask first
            valid_scores = raw_scores[valid_mask]
            valid_boxes = raw_boxes[valid_mask]
            valid_labels = raw_labels[valid_mask]
            
            # Apply confidence threshold
            conf_mask = valid_scores > confidence_threshold
            
            if conf_mask.sum() == 0:
                return [[], [], []]
            
            # Apply confidence mask to get final results
            final_boxes = valid_boxes[conf_mask].cpu().numpy().tolist()
            final_scores = valid_scores[conf_mask].cpu().numpy().tolist()
            final_classes = valid_labels[conf_mask].cpu().numpy().tolist()
            
            return [final_boxes, final_scores, final_classes]
        
        # Handle other model formats (like YOLO results)
        elif hasattr(output, 'boxes') or (isinstance(output, list) and len(output) > 0 and hasattr(output[0], 'boxes')):
            # YOLO-style results
            if isinstance(output, list):
                results = output[0]  # Take first result
            else:
                results = output
                
            if not hasattr(results, 'boxes') or results.boxes is None:
                return [[], [], []]
                
            boxes = results.boxes.xyxy.cpu().numpy().tolist()
            scores = results.boxes.conf.cpu().numpy().tolist()
            classes = results.boxes.cls.cpu().numpy().tolist()
            
            # Apply confidence threshold
            filtered_boxes = []
            filtered_scores = []
            filtered_classes = []
            
            for box, score, cls in zip(boxes, scores, classes):
                if score > confidence_threshold:
                    filtered_boxes.append(box)
                    filtered_scores.append(score)
                    filtered_classes.append(cls)
            
            return [filtered_boxes, filtered_scores, filtered_classes]
        
        else:
            # Unknown format, return empty
            print(f"Unknown output format: {type(output)}")
            return [[], [], []]
            
    except Exception as e:
        print(f"Error in postprocessing: {e}")
        return [[], [], []]


def build_engine_from_onnx(onnx_path, engine_path, workspace_size=(2 << 60)):
    """Build TensorRT engine from ONNX file using the build_trt functionality"""
    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT is not available. Cannot build engine from ONNX.")
    
    try:
        # Import the build engine function
        from build_trt import build_engine
        
        print(f"Building TensorRT engine...")
        print(f"  ONNX: {onnx_path}")
        print(f"  Engine: {engine_path}")
        print(f"  Workspace: {workspace_size // (1024**3)}GB")
        
        # Build the engine with reduced workspace to avoid memory issues
        build_engine(onnx_path, engine_path, workspace_size)
        
        print(f"TensorRT engine built successfully: {engine_path}")
        
    except ImportError as e:
        raise ImportError(f"Cannot import build_trt module: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to build TensorRT engine: {e}")

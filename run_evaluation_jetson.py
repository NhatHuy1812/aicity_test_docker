import os
import time
import argparse
import cv2
import json
import numpy as np

from utils import f1_score

from utils import get_model , preprocess_image , postprocess_result, changeId, is_grayscale
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='/data/FishEye1K_eval/images', help='Path to image folder')
    parser.add_argument('--day_model_path', type=str, default='/models/day_m.onnx', help='Path to the day model')
    parser.add_argument('--night_model_path', type=str, default='/models/night_m.onnx', help='Path to the night model')
    parser.add_argument('--max_fps', type=float, default=25.0, help='Maximum FPS for evaluation')
    parser.add_argument('--output_json', type=str, default='/data/FishEye1K_eval/predictions.json', help='Output JSON file for predictions')
    parser.add_argument('--ground_truths_path', type=str, default='/data/FishEye1K_eval/groundtruth.json', help='Path to ground truths JSON file')
    parser.add_argument('--confidence_threshold', type=float, default=0.65, help='Confidence threshold for detections')
    args = parser.parse_args()

    image_folder = args.image_folder
    day_model_path = args.day_model_path
    night_model_path = args.night_model_path

    # Load both models
    print(f"Loading day model: {day_model_path}")
    day_model = get_model(day_model_path)
    print(f"Loading night model: {night_model_path}")
    night_model = get_model(night_model_path)

    image_files = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    print(f"Found {len(image_files)} images.")

    predictions = []
    print('Prediction started')
    day_count = 0
    night_count = 0
    # preprocess_times = []
    # inference_times = []
    # postprocess_times = []
    # total_times = []
    # preprocess_time = 0
    # inference_time = 0
    # postprocess_time = 0
    total_time = 0
    start_time = time.time()
    for image_path in image_files:
        img = cv2.imread(image_path)
        if img is None:
            # print(f"Warning: Could not read image {image_path}. Skipping.")
            continue
        t0=time.time()
        img = preprocess_image(img)
        
        # Determine if image is night (grayscale) or day (color)
        try:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            if "_N_" in image_name:
                is_night = True
            elif "_E_" in image_name:
                is_night = True
            elif "_A_" in image_name:
                is_night = False
            elif "_M_" in image_name:
                is_night = False
            else:
                is_night = is_grayscale(img, 0.05)
        except:
            is_night = is_grayscale(img, 0.05)
        
        # Use appropriate model
        if is_night:
            results = night_model(img, verbose=False)
            night_count += 1
        else:
            results = day_model(img, verbose=False)
            day_count += 1
        
        # t2 = time.time()
        results = postprocess_result(results, args.confidence_threshold) # [boxes, scores, classes]
        predictions.append((image_path,results))
        t3 = time.time()

        #print(f"Processed {os.path.basename(image_path)}: {len(results[0])} objects detected.")
        #preprocess_time += (t1-t0)
        #inference_time  += ( t2-t1)
        #postprocess_time += (t3-t2)
        total_time += (t3-t0)
        # preprocess_times.append(t1-t0)
        # inference_times.append(t2-t1)
        # postprocess_times.append(t3-t2)
        # total_times.append(t3-t0)
        # print(f"Preprocessing Time   : {(t1 - t0)*1000:.2f} ms")
        # print(f"Inference Time       : {(t2 - t1)*1000:.2f} ms")
        # print(f"Postprocessing Time  : {(t3 - t2)*1000:.2f} ms")
        # print(f"Total Time           : {(t3 - t0)*1000:.2f} ms")
        # break
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processed {len(image_files)} images in {elapsed_time:.2f} seconds.")
    print(f"Day model used for {day_count} images, Night model used for {night_count} images.")

    # print(f"Avg Image Preprocess Time      : {preprocess_time/len(image_files)*1000:.2f} ms")
    # print(f"Avg Inference Time       : {inference_time/len(image_files)*1000:.2f} ms")
    # print(f"Avg Postprocessing Time  : {postprocess_time/len(image_files)*1000:.2f} ms")
    print(f"Avg Processing Time           : {total_time/len(image_files)*1000:.2f} ms")

    # with open('/data/yolo11n_tensorrt_preprocess_times.txt', 'w') as f:
    #     for t in preprocess_times:
    #         f.write(f"{t}\n")

    # with open('/data/yolo11n_tensorrt_inference_times.txt', 'w') as f:
    #     for t in inference_times:
    #         f.write(f"{t}\n")

    # with open('/data/yolo11n_tensorrt_postprocess_times.txt', 'w') as f:
    #     for t in postprocess_times:
    #         f.write(f"{t}\n")

    # with open('/data/yolo11n_tensorrt_total_times.txt', 'w') as f:
    #     for t in total_times:
    #         f.write(f"{t}\n")
    predictions_json = []
    
    # Convert predictions to COCO format
    for image_path, results in predictions:
        image_name = os.path.basename(image_path)
        image_id = changeId(os.path.splitext(image_name)[0])
        
        boxes, scores, classes = results
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            predictions_json.append({
                "image_id": image_id,
                "category_id": int(cls),
                "bbox": [x1, y1, width, height],
                "score": float(score)
            })
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    
    # Save predictions to JSON
    with open(args.output_json, 'w') as f:
        json.dump(predictions_json, f, indent=2)
    
    print(f"Saved {len(predictions_json)} detections to {args.output_json}")

    fps = len(image_files) / total_time
    normfps = min(fps, args.max_fps)/args.max_fps

    f1 = f1_score(args.output_json, args.ground_truths_path)
    harmonic_mean = 2 * f1 * normfps / (f1 + normfps)

    print(f"\n--- Evaluation Complete ---")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"FPS: {fps:.2f}")
    print(f"Normalized FPS: {normfps:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Metric (harmonic mean of F1-score and NormalizedFPS): {harmonic_mean:.4f}")

if __name__ == "__main__":
    main()

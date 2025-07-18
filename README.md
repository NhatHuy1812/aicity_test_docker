Sắp hết bộ nhớ … Nếu hết bộ nhớ, thì bạn không thể lưu tệp vào Drive, sao lưu vào Google Photos hoặc sử dụng Gmail.
Dưới đây là một bản README tiếng Anh mà bạn có thể dùng để hướng dẫn người dùng cách chạy dự án của bạn, bao gồm cách tải Docker image, cấu trúc dữ liệu, cách chạy, và cách đánh giá:

---

# FishEye1K Evaluation with TensorRT Models

This project evaluates object detection performance using pre-built TensorRT engines for both day and night images.

## Docker Setup

First, load the Docker image from the `run_test.tar` file [download here](https://drive.google.com/file/d/17IdghKII7Cnjd8ILbMkCqyUiwxBxyll8/view?pli=1):

```bash
docker load -i run_test.tar
```

### Run the Container

Set the environment variable `DATA_DIR` to the path of your dataset directory and run the container:

```bash
IMAGE="hieupham1103/aicity_dfine:latest"
DATA_DIR="/data"  # change this to your local path

docker run -it --ipc=host --runtime=nvidia -v ${DATA_DIR}:/data ${IMAGE}
```

This command mounts the host's dataset directory into the container and enables GPU acceleration.

---

## Dataset Structure

Your dataset directory should have the following structure:

```
/data/
└── FishEye1K_eval/
    ├── images/              # Input images (.jpg, .png, etc.)
    ├── groundtruth.json     # Ground truth annotations in COCO format
    └── predictions.json     # Output file (auto-generated after evaluation)
```

---

## What the Code Does

1. **Load ONNX Models**:

   * The script loads two `.onnx` models.

2. **Build TensorRT Engines**:

   * Each ONNX model is converted into a TensorRT engine **at runtime**.
   * This step may take a few seconds depending on your Jetson hardware.

3. **Infer Image Type**: It determines whether each image is a day or night image based on:

   * filename patterns (`_N_`, `_E_`, `_A_`, `_M_`)
   * or grayscale detection fallback

4. **Inference Pipeline**:

   * Preprocess the image
   * Run inference using the appropriate model
   * Post-process results and collect predictions

5. **Output**:

   * The results are saved to `/data/FishEye1K_eval/predictions.json` in COCO format.

6. **Metrics**:

   * FPS (Frames Per Second)
   * Normalized FPS = `min(FPS, max_fps) / max_fps`
   * F1 Score (compared with ground truth)
   * Harmonic mean between F1 Score and Normalized FPS

---

## Time Measurement

For each image, the following processing times are measured:

* **Preprocessing**
* **Inference**
* **Postprocessing**
* **Total Time**

At the end of the run, the script reports:

* Total time elapsed
* Average time per image
* Frames per second (FPS)
* Normalized FPS
* Final F1 Score
* **Harmonic mean** = `2 * (F1 * NormalizedFPS) / (F1 + NormalizedFPS)`

This metric balances both accuracy and speed for real-time evaluation.

---

## Sample Output

```
Found 100 images.
Prediction started
Processed 100 images in 12.35 seconds.
Day model used for 58 images, Night model used for 42 images.
Avg Processing Time: 98.20 ms
Saved 1000 detections to /data/FishEye1K_eval/predictions.json

--- Evaluation Complete ---
Total time: 12.35 seconds
Total processing time: 9.82 seconds
FPS: 10.18
Normalized FPS: 0.4072
F1-score: 0.8413
Metric (harmonic mean of F1-score and NormalizedFPS): 0.5486
```

---

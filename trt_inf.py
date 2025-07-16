"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import collections
import contextlib
import os
import time
from collections import OrderedDict

# Fix numpy compatibility issues before importing other packages
import numpy as np
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

import cv2  # Added for video processing
import tensorrt as trt
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
import onnxruntime as ort  # Add ONNX runtime import


class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self):
        self.total = 0

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start

    def reset(self):
        self.total = 0

    def time(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()


class TRTInference(object):
    def __init__(
        self, engine_path, device="cuda:0", backend="torch", max_batch_size=32, verbose=False
    ):
        self.engine_path = engine_path
        self.device = device
        self.backend = backend
        self.max_batch_size = max_batch_size

        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)

        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        

        
        self.bindings = self.get_bindings(
            self.engine, self.context, self.max_batch_size, self.device
        )
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())
        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        self.time_profile = TimeProfiler()

    def load_engine(self, path):
        trt.init_libnvinfer_plugins(self.logger, "")
        with open(path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def get_input_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def get_output_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(self, engine, context, max_batch_size=32, device=None) -> OrderedDict:
        Binding = collections.namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        bindings = OrderedDict()

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if shape[0] == -1:
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    context.set_input_shape(name, shape)

            # Convert shape to tuple for torch.empty
            shape_tuple = tuple(shape)

            try:
                # Try to create numpy array first
                np_array = np.empty(shape_tuple, dtype=dtype)
                data = torch.from_numpy(np_array).to(device)
            except Exception as e:
                print(f"NumPy fallback for {name}: {e}")
                # Fallback: create tensor directly without numpy
                try:
                    if dtype == np.float32:
                        data = torch.empty(shape_tuple, dtype=torch.float32).to(device)
                    elif dtype == np.float16:
                        data = torch.empty(shape_tuple, dtype=torch.float16).to(device)
                    else:
                        data = torch.empty(shape_tuple, dtype=torch.float32).to(device)
                except Exception as cuda_e:
                    print(f"CUDA fallback to CPU for {name}: {cuda_e}")
                    # Final fallback to CPU
                    if dtype == np.float32:
                        data = torch.empty(shape_tuple, dtype=torch.float32, device='cpu')
                    elif dtype == np.float16:
                        data = torch.empty(shape_tuple, dtype=torch.float16, device='cpu')
                    else:
                        data = torch.empty(shape_tuple, dtype=torch.float32, device='cpu')
            
            bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

        return bindings

    def run_torch(self, blob):
        # Copy input data to bindings
        for n in self.input_names:
            input_data = blob[n]

            # Ensure data is on the right device and dtype
            if input_data.dtype != self.bindings[n].data.dtype:
                input_data = input_data.to(dtype=self.bindings[n].data.dtype)
            
            # Handle dynamic shapes
            binding_shape = tuple(self.bindings[n].shape)
            input_shape = tuple(input_data.shape)
            
            if binding_shape != input_shape:
                
                # Check if this is a dynamic input
                engine_shape = self.engine.get_tensor_shape(n)
                
                if -1 in engine_shape:
                    try:
                        self.context.set_input_shape(n, input_data.shape)
                        
                        # Recreate binding with new shape
                        shape_tuple = tuple(input_data.shape)
                        if input_data.device.type == 'cuda':
                            data = torch.empty(shape_tuple, dtype=self.bindings[n].data.dtype, device=input_data.device)
                        else:
                            data = torch.empty(shape_tuple, dtype=self.bindings[n].data.dtype, device='cpu')
                        self.bindings[n] = self.bindings[n]._replace(shape=input_data.shape, data=data)
                        self.bindings_addr[n] = data.data_ptr()
                    except Exception as e:
                        return self._create_dummy_outputs()
                else:
                    return self._create_dummy_outputs()
            
            # Copy input data to binding
            try:
                self.bindings[n].data.copy_(input_data)
            except Exception as e:
                return self._create_dummy_outputs()
        
        # Execute inference
        try:
            success = self.context.execute_v2(list(self.bindings_addr.values()))
            if not success:
                return self._create_dummy_outputs()
        except Exception as e:
            return self._create_dummy_outputs()
        
        # Get outputs
        outputs = {}
        for n in self.output_names:
            outputs[n] = self.bindings[n].data.clone()
        
        return outputs
    
    def _create_dummy_outputs(self):
        """Create dummy outputs when inference fails"""
        dummy_outputs = {}
        for n in self.output_names:
            dummy_outputs[n] = torch.zeros_like(self.bindings[n].data)
        return dummy_outputs

    def __call__(self, blob):
        if self.backend == "torch":
            return self.run_torch(blob)
        else:
            raise NotImplementedError("Only 'torch' backend is implemented.")

    def synchronize(self):
        if self.backend == "torch" and torch.cuda.is_available():
            torch.cuda.synchronize()


class ONNXInference(object):
    def __init__(self, onnx_path, device='cpu'):
        self.onnx_path = onnx_path
        self.device = device
        # Choose execution providers based on device
        providers = ['CPUExecutionProvider']
        if 'cuda' in device.lower():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def __call__(self, blob):
        # Prepare numpy inputs for ONNX session
        np_inputs = {}
        for name in self.input_names:
            tensor = blob.get(name)
            if tensor is None:
                raise KeyError(f"Missing input '{name}' for ONNX model")
            np_inputs[name] = tensor.cpu().numpy()
        # Run inference
        np_outputs = self.session.run(self.output_names, np_inputs)
        # Convert outputs back to torch tensors on desired device
        torch_outputs = {}
        for name, out in zip(self.output_names, np_outputs):
            torch_outputs[name] = torch.from_numpy(out).to(self.device)
        return torch_outputs


def draw(images, labels, boxes, scores, thrh=0.4):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline="red")
            draw.text(
                (b[0], b[1]),
                text=f"{lab[j].item()} {round(scrs[j].item(), 2)}",
                fill="blue",
            )

    return images


def process_image(m, file_path, device):
    try:
        im_pil = Image.open(file_path).convert("RGB")
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(device)

        transforms = T.Compose(
            [
                T.Resize((1280, 1280)),
                T.ToTensor(),
            ]
        )
        im_data = transforms(im_pil)[None]

        blob = {
            "images": im_data.to(device),
            "orig_target_sizes": orig_size.to(device),
        }

        # Measure inference time
        start_time = time.time()
        output = m(blob)
        end_time = time.time()
        inference_time = end_time - start_time
        
        print(f"Inference time: {inference_time:.4f} seconds ({1/inference_time:.2f} FPS)")
        # print(output)
        # Check for valid detections before drawing
        valid_scores = output["scores"][~torch.isnan(output["scores"])]
        valid_boxes = output["boxes"][~torch.isnan(output["boxes"]).any(dim=-1)]
        
        if len(valid_scores) == 0:
            result_images = [im_pil]
        else:
            result_images = draw([im_pil], output["labels"], output["boxes"], output["scores"], thrh=0.59)
        
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Save with original filename
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        result_path = f"results/result_{base_name}.jpg"
        result_images[0].save(result_path)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Still try to save a blank result to show the error
        os.makedirs("results", exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        error_path = f"results/error_{base_name}.txt"
        with open(error_path, 'w') as f:
            f.write(f"Error processing {file_path}: {e}")
        print(f"Error log saved as '{error_path}'.")


def process_video(m, file_path, device):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("trt_result.mp4", fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose(
        [
            T.Resize((1280, 1280)),
            T.ToTensor(),
        ]
    )

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([w, h])[None].to(device)

        im_data = transforms(frame_pil)[None]

        blob = {
            "images": im_data.to(device),
            "orig_target_sizes": orig_size.to(device),
        }

        output = m(blob)

        # Draw detections on the frame
        result_images = draw([frame_pil], output["labels"], output["boxes"], output["scores"])

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(result_images[0]), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()


def process_folder(m, folder_path, device):
    """Process all images in a folder"""
    if not os.path.isdir(folder_path):
        return
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(image_extensions):
            image_files.append(os.path.join(folder_path, file))
    
    if not image_files:
        return
    
    # Process each image
    for i, image_path in enumerate(image_files):
        try:
            im_pil = Image.open(image_path).convert("RGB")
            w, h = im_pil.size
            orig_size = torch.tensor([w, h])[None].to(device)

            transforms = T.Compose([
                T.Resize((640, 640)),
                T.ToTensor(),
            ])
            im_data = transforms(im_pil)[None]

            blob = {
                "images": im_data.to(device),
                "orig_target_sizes": orig_size.to(device),
            }

            output = m(blob)
            result_images = draw([im_pil], output["labels"], output["boxes"], output["scores"])
            
            # Create results directory if it doesn't exist
            os.makedirs("results", exist_ok=True)
            
            # Save result with original filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            result_path = f"results/result_{base_name}.jpg"
            result_images[0].save(result_path)
            
        except Exception as e:
            # Save error log
            os.makedirs("results", exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            error_path = f"results/error_{base_name}.txt"
            with open(error_path, 'w') as f:
                f.write(f"Error processing {image_path}: {e}")
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-trt", "--trt", type=str, required=True)
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-d", "--device", type=str, default="cuda:0")

    args = parser.parse_args()

    # Select appropriate inference class based on file extension
    if args.trt.lower().endswith('.onnx'):
        m = ONNXInference(args.trt, device=args.device)
    else:
        m = TRTInference(args.trt, device=args.device)

    file_path = args.input
    
    if os.path.isdir(file_path):
        # Process folder
        process_folder(m, file_path, args.device)
    elif os.path.splitext(file_path)[-1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        # Process as image
        process_image(m, file_path, args.device)
    else:
        # Process as video
        process_video(m, file_path, args.device)

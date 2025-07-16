FROM nvcr.io/nvidia/l4t-jetpack:r35.4.1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install numpy first with compatible version
RUN pip3 install --no-cache-dir "numpy==1.19.5"

# Install PyTorch with CUDA support for Jetson
RUN pip3 install --no-cache-dir \
    https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl

# Install torchvision
RUN pip3 install --no-cache-dir "torchvision==0.13.0"

# Install other packages
RUN pip3 install --no-cache-dir \
    "pillow==8.3.2" \
    "opencv-python==4.5.3.56" \
    "onnx==1.11.0" \
    "onnxruntime"

# RUN pip3 install pycocotools
RUN pip3 install --no-cache-dir "matplotlib==3.3.4"

RUN pip3 install --no-cache-dir cython

COPY cocoapi cocoapi
RUN pip3 install -e /app/cocoapi/PythonAPI
# RUN cd /app/cocoapi/PythonAPI && \
#     python3 setup.py build_ext install && \
# RUN rm -rf /app/cocoapi/

ENV PATH="/usr/src/tensorrt/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/tegra:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Copy model files and inference code
# COPY dfine_l_day.engine /workspace/dfine_l_day.engine
COPY trt_inf.py trt_inf.py
COPY run_evaluation_jetson.py run_evaluation_jetson.py
COPY utils.py utils.py
COPY pycocotools pycocotools_mod
COPY build_trt.py build_trt.py
COPY ./models /models/
# Copy evaluation data (if available)
# COPY Fisheye1K_eval /workspace/Fisheye1K_eval

# Set shell and make evaluation script executable
# SHELL ["/bin/bash", "-lc"]
RUN chmod +x run_evaluation_jetson.py

# Entry point for evaluation
# ENTRYPOINT ["bash"]
ENTRYPOINT ["python3", "run_evaluation_jetson.py"]

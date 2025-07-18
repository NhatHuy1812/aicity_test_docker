# IMAGE="hieupham1103/aicity_dfine:dual"
# DATA_DIR="/home/becamex/FishEye8K/challenge_iccv_2025_hieu/data"
# MODEL_DIR="/home/becamex/FishEye8K/temp"
# docker run -it --ipc=host --runtime=nvidia \
#   -v ${DATA_DIR}:/data \
#   -v ${MODEL_DIR}:/models \
#   ${IMAGE} \
#   --confidence_threshold 0.50

IMAGE="hieupham1103/aicity_dfine:latest"
DATA_DIR="/home/becamex/FishEye8K/challenge_iccv_2025_hieu/data"

docker run -it --ipc=host --runtime=nvidia -v ${DATA_DIR}:/data ${IMAGE}
FROM python:3.8.10-slim  
  
ARG DEBIAN_FRONTEND=noninteractive  
ARG TARGETARCH  
  
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 vim ffmpeg zip unzip htop \
    screen tree build-essential gcc g++ python3-dev libatlas-base-dev make libprotobuf-dev protobuf-compiler && apt-get clean && rm -rf /var/lib/apt/lists  
  
RUN pip install --upgrade pip  
  
# 將 requirements.txt 複製到 Docker 映像中
COPY requirements.txt .  
COPY whl/torch-1.7.0+cu101-cp38-cp38-linux_x86_64.whl .
COPY whl/mmcv_full-1.3.17-cp38-cp38-manylinux1_x86_64.whl .
COPY whl/mmdet-2.25.0-py3-none-any.whl .
  
# 安裝 python packages  
RUN pip3 install numpy  
RUN pip3 install -r requirements.txt  
RUN pip3 install lap  
  
# 設置工作目錄  
WORKDIR /app  
  
# 复制 app 资料夹到 Docker 映像中的 /app 目录  
COPY . /app  
  
# 设置环境变量  
ENV LC_ALL=C.UTF-8  
ENV LANG=C.UTF-8  
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility  
ENV NVIDIA_VISIBLE_DEVICES=all  
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64  
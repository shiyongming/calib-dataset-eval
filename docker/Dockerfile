FROM nvcr.io/nvidia/tensorrt:20.11-py3

### update apt and install libs
RUN apt-get update &&\
    apt-get install -y vim cmake libsm6 libxext6 libxrender-dev libgl1-mesa-glx git

### create folder
RUN mkdir ~/workspace &&\
    mkdir /root/.pip

### set pip source
#COPY ./pip.conf /root/.pip
#COPY pip.conf /etc/pip.conf

### python
RUN pip3 install --upgrade pip

### pytorch
RUN pip3 install torch==1.7.0+cu110 torchvision==0.8.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

### install mmcv
#ADD sources.list /etc/apt/
#RUN pip install mmcv-full==1.2.0+torch1.7.0+cu110 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
RUN pip install mmcv-full==1.2.4 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

### git mmdetection
RUN git clone --depth=1 https://github.com/open-mmlab/mmdetection.git /root/workspace/mmdetection

### install mmdetection
RUN cd /root/workspace/mmdetection &&\ 
    pip3 install -r requirements.txt &&\
    python3 setup.py develop

### git amirstan plugin
RUN git clone --depth=1 https://github.com/grimoire/amirstan_plugin.git /root/workspace/amirstan_plugin &&\ 
    cd /root/workspace/amirstan_plugin &&\ 
    git submodule update --init --progress --depth=1

### install amirstan plugin
RUN cd /root/workspace/amirstan_plugin &&\ 
    mkdir build &&\
    cd build &&\
    cmake .. &&\
    make -j10

ENV AMIRSTAN_LIBRARY_PATH=/root/workspace/amirstan_plugin/build/lib

### git torch2trt_dynamic
RUN git clone --depth=1 https://github.com/grimoire/torch2trt_dynamic.git /root/workspace/torch2trt_dynamic

### install torch2trt_dynamic
RUN cd /root/workspace/torch2trt_dynamic &&\
    python3 setup.py develop

### git mmdetection-to-tensorrt
RUN git clone --depth=1 https://github.com/grimoire/mmdetection-to-tensorrt.git /root/workspace/mmdetection-to-tensorrt

### install mmdetection-to-tensorrt
RUN cd /root/workspace/mmdetection-to-tensorrt &&\
    python3 setup.py develop

### git calib-dataset-eval
RUN git clone --depth=1 https://github.com/shiyongming/calib-dataset-eval.git /root/workspace/calib-dataset-eval

### install mmdetection-to-tensorrt
RUN cd /root/workspace/calib-dataset-eval &&\
    python3 setup.py develop

WORKDIR /root/workspace

CMD [ "--help" ]
ENTRYPOINT [ "bin/ls" ]
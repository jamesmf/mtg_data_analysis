FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

COPY ./requirements.txt requirements.txt

RUN apt-get update && apt-get install -y software-properties-common  && add-apt-repository ppa:deadsnakes/ppa && apt-get update

RUN DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt-get -y install tzdata python3.9 python3.9-distutils python3.9-dev curl build-essential pkg-config

RUN curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.9 get-pip.py && rm get-pip.py

RUN apt-get install -y graphviz graphviz-dev git  

RUN python3.9 -m pip install -r requirements.txt 

RUN python3.9 -m pip install dgl-cu113 -f https://data.dgl.ai/wheels/repo.html &&  python3.9 -m pip install torch==1.10.0+cu113 pytorch-lightning==1.4.9 -f https://download.pytorch.org/whl/cu113/torch_stable.html 
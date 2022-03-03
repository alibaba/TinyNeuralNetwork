FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
COPY requirements.txt /

RUN apt-get update
RUN apt-get install -y wget git && rm -rf /var/lib/apt/lists/*
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py37_4.10.3-Linux-x86_64.sh

RUN conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
RUN pip install -r requirements.txt

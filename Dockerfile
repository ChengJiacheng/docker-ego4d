# FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
# FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
# FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
# FROM nvidia/cuda:11.1.1-runtime-ubuntu18.04
# FROM nvcr.io/nvidia/pytorch:21.06-py3
# FROM nvcr.io/nvidia/pytorch:20.12-py3
# FROM ubuntu:18.04
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive


RUN apt-get update && apt-get install -y libglib2.0-0 libxrender1 libsm6 libxext6 libxrender-dev vim wget sudo psmisc locales cmake vim g++ zip \
     htop git screen git-lfs gnupg libgl1 bwm-ng iputils-ping dnsutils curl sysstat axel rsync
# nvidia-cuda-toolkit
RUN locale-gen en_US.UTF-8

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
    
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH


# ADD https://raw.githubusercontent.com/ChengJiacheng/nautilus/master/yaml/torch19.yml /tmp/environment.yml
# RUN conda env create --file /tmp/environment.yml && conda init bash && echo "source activate base" >> ~/.bashrc
RUN conda create -n torch python=3.8 numpy=1.23.5 -y && conda init bash && echo "source activate torch" >> ~/.bashrc

ENV PATH /opt/conda/envs/torch/bin:$PATH


# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
# RUN sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
# RUN wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda-repo-ubuntu1804-11-1-local_11.1.0-455.23.05-1_amd64.deb
# RUN sudo dpkg -i cuda-repo-ubuntu1804-11-1-local_11.1.0-455.23.05-1_amd64.deb
# RUN sudo apt-key add /var/cuda-repo-ubuntu1804-11-1-local/7fa2af80.pub
# RUN sudo apt-get update
# RUN sudo apt-get -y install cuda

# RUN apt-get update && apt-get install -y libopenblas-dev

# RUN yes | pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
# RUN conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113


# RUN conda install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia


# RUN conda install -y -c nvidia/label/cuda-11.3.1 cuda-nvcc
# ADD https://github.com/cdr/code-server/releases/download/3.0.1/code-server-3.0.1-linux-x86_64.tar.gz ./ 
# RUN tar -xvzf code-server-3.0.1-linux-x86_64.tar.gz && rm -r code-server-3.0.1-linux-x86_64.tar.gz && code-server-3.0.1-linux-x86_64/code-server --install-extension ms-python.python

# ADD https://github.com/coder/code-server/releases/download/v4.0.2/code-server-4.0.2-linux-amd64.tar.gz ./ 
# RUN tar -xvzf code-server-4.0.2-linux-amd64.tar.gz --remove-files && code-server-4.0.2-linux-amd64/code-server --install-extension ms-python.python

# ADD https://github.com/coder/code-server/releases/download/v4.3.0/code-server_4.3.0_amd64.deb ./ 
# RUN sudo dpkg -i code-server_4.3.0_amd64.deb && code-server --install-extension ms-python.python

# ADD https://github.com/coder/code-server/releases/download/v4.5.1/code-server_4.5.1_amd64.deb ./ 
# RUN sudo dpkg -i code-server_4.5.1_amd64.deb && code-server --install-extension ms-python.python

# ADD https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_2.1.1_linux_386.tar.gz ./ 
# RUN tar xvf gdrive_2.1.1_linux_386.tar.gz && chmod +x gdrive && sudo install gdrive /usr/local/bin/gdrive

# ADD https://secure.nic.cz/files/knot-resolver/knot-resolver-release.deb ./ 
# RUN sudo dpkg -i knot-resolver-release.deb && sudo apt update && sudo apt install -y knot-resolver 

# RUN conda install -y scikit-learn pandas matplotlib jupyter gpustat cython

# RUN yes | pip3 install scikit-learn pandas matplotlib jupyter gpustat cython opencv-python gdown runstats && \
#     yes | pip3 install img2dataset && \
#     yes | pip3 install pycocotools tensorboard tensorboardX yacs json_tricks xtcocotools 

RUN yes | pip3 install xtcocotools wandb ipykernel torch-tb-profiler

ADD https://raw.githubusercontent.com/TexasInstruments/edgeai-yolov5/yolo-pose/requirements.txt ./ 
RUN pip install -r requirements.txt

RUN git clone https://github.com/Jeff-sjtu/CrowdPose.git && cd CrowdPose/crowdpose-api/PythonAPI/ && make install && python setup.py install --user

# RUN pip install --upgrade scipy==1.8.1

RUN pip install ftfy regex tqdm git+https://github.com/openai/CLIP.git decord

RUN sudo -v ; curl https://rclone.org/install.sh | sudo bash

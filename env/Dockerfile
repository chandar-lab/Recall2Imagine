FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git xvfb curl \
    libglu1-mesa libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev mesa-utils freeglut3 freeglut3-dev \
    libglew2.1 libglfw3 libglfw3-dev libegl-dev zlib1g zlib1g-dev libsdl2-dev libjpeg-dev lua5.1 liblua5.1-0-dev libffi-dev \
    build-essential cmake g++ build-essential pkg-config software-properties-common gettext \
    ffmpeg patchelf swig unrar unzip zip curl wget tmux \
    && rm -rf /var/lib/apt/lists/*


ENV CNDA=/conda
RUN mkdir -p $CNDA && chmod 755 $CNDA
RUN curl -Lo $CNDA/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
 && chmod +x $CNDA/miniconda.sh \
 && $CNDA/miniconda.sh -b -p $CNDA/miniconda \
 && rm $CNDA/miniconda.sh
ENV PATH=$CNDA/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.8 environment
RUN $CNDA/miniconda/bin/conda create -y --name py38 python=3.8 \
 && $CNDA/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py38
ENV CONDA_PREFIX=$CNDA/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN $CNDA/miniconda/bin/conda clean -ya
# Create conda environment
# Atari
RUN pip3 install --upgrade setuptools pip
ADD ./gym-0.19.0-py3-none-any.whl /root
RUN pip3 install \
    tensorflow_probability==0.20.1 \
    yacs==0.1.8 \
    tensorflow-cpu==2.12.0 \
    ruamel.yaml==0.17.32 \
    ruamel.yaml.clib==0.2.7 \
    moviepy==1.0.3 \
    imageio==2.31.1 \
    crafter==1.8.1 \
    dm-control==1.0.12 \
    mujoco==2.3.6 \
    robodesk==1.0.0 \
    bsuite==0.3.5 \
    numpy==1.22.1 \
    opt_einsum==3.3.0 \
    einops==0.6.1 \
    wandb==0.15.5 \
    memory-maze==1.0.2 \
    popgym==1.0.2 \
    gymnasium==0.29.0 \
    mazelib==0.9.13 \
    procgen==0.10.7 \
    atari-py==0.2.9 \
    protobuf==3.20.3 \
    gast==0.4.0 \
    zmq \
    /root/gym-0.19.0-py3-none-any.whl \ 
    rich==13.7.0 \
    msgpack==1.0.7 \
    cloudpickle==1.6.0 \
    opencv-python==4.8.0.74

RUN pip3 install --upgrade \
    flax \
    dm-haiku \
    "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    optax==0.1.5

RUN wget -L -nv http://www.atarimania.com/roms/Roms.rar && \
    unrar x -y Roms.rar && \
    python3 -m atari_py.import_roms ROMS && \
    rm -rf Roms.rar ROMS.zip ROMS
RUN mkdir -p /root/.mujoco && \
    cd /root/.mujoco && \
    wget -nv https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz && \
    tar -xf mujoco.tar.gz && \
    rm mujoco.tar.gz

ENV OMP_NUM_THREADS 1
ENV PYTHONUNBUFFERED 1
ENV LANG "C.UTF-8"
ENV NUMBA_CACHE_DIR=/tmp
ENV XLA_PYTHON_CLIENT_MEM_FRACTION 0.8
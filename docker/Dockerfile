# Select one the following parent images based on your preferred ubuntu
# distribution and python version.
#
# All these images have been conveniently built to include all the necessary
# dependencies needed to run the challenge. 
# FROM lac-leaderboard:ubuntu20.04-py3.8
# FROM lac-leaderboard:ubuntu20.04-py3.9
FROM lac-leaderboard:ubuntu22.04-py3.10

# Install any system dependencies needed by your agent.
# RUN apt-get update && apt-get install -y \
#         any-system-dependency \
#     && rm -rf /var/lib/apt/lists/*
# RUN apt-get update && apt-get install -y git

# Install system dependencies for ORBSLAM3
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    libopencv-dev \
    libeigen3-dev \
    libgl1-mesa-dev \ 
    libegl1-mesa-dev \
    libepoxy-dev \
    libglew-dev \ 
    libjpeg-dev \ 
    libpng-dev \
    catch2 \
    libavcodec-dev \ 
    libavutil-dev \ 
    libavformat-dev \ 
    libswscale-dev \
    libavdevice-dev \
    python3-dev \
    python3-pip \
    python3-pybind11 \
    libboost-all-dev \
    libssl-dev \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Install python library required by your agent.
RUN python3 -m pip install --upgrade pip && pip install wheel uv

# Setup SSH access to github
# FOR TESTING ONLY
RUN mkdir -p /root/.ssh && ssh-keyscan github.com >> /root/.ssh/known_hosts
COPY team_code/github_ssh_key /root/.ssh/id_ed25519
RUN chmod 600 /root/.ssh/id_ed25519

# Install major python dependencies so that this layer is cached and reused
RUN uv pip install \
    "numpy==1.21.4" \
    "opencv-python==4.10.0.84" \
    "pyapriltags==3.3.0.3" \
    "pytransform3d==3.13.0" \
    "scikit-learn==1.6.1" \
    "scipy==1.10.1" \
    "setuptools==75.8.0" \
    "shapely==2.0.7" \
    "torch==2.5.1" \  
    "ultralytics==8.0.120" \
    "git+https://github.com/CASIA-IVA-Lab/FastSAM@b4ed20c" \
    "git+ssh://git@github.com/Robaire/orbslam3-python@07ed5f2"

# DO NOT EDIT THIS LINE!!!
# This command copies your agent's code into the correct place in the container.
ARG CACHE_BREAKER
ENV TEAM_CODE_ROOT=/workspace/team_code
RUN echo ${CACHE_BREAKER}
COPY team_code ${TEAM_CODE_ROOT}

# Install MAPLE dependencies
RUN uv pip install -r ${TEAM_CODE_ROOT}/pyproject.toml

# Point to your agent's entrypoint file and configuration file
# ONLY MODIFY <AGENT_ENTRYPOINT_FILENAME> and <CONFIG_ENTRYPOINT_FILENAME>
# placeholders. These filenames need to be relative to your agent's team code
# folder.
ENV TEAM_AGENT=${TEAM_CODE_ROOT}/mit_agent.py
ENV TEAM_CONFIG=${TEAM_CODE_ROOT}/

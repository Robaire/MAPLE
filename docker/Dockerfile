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
RUN apt-get update && apt-get install -y git

# Install system dependencies for ORBSLAM3
RUN RUN apt-get install -y \
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
    libssl-dev

# DO NOT EDIT THIS LINE!!!
# This command copies your agent's code into the correct place in the container.
ENV TEAM_CODE_ROOT=/workspace/team_code
COPY team_code ${TEAM_CODE_ROOT}

# Install python library required by your agent.
RUN python3 -m pip install --upgrade pip && pip install wheel uv

# Install MAPLE dependencies
RUN uv pip install -r ${TEAM_CODE_ROOT}/pyproject.toml

# Point to your agent's entrypoint file and configuration file
# ONLY MODIFY <AGENT_ENTRYPOINT_FILENAME> and <CONFIG_ENTRYPOINT_FILENAME>
# placeholders. These filenames need to be relative to your agent's team code
# folder.
ENV TEAM_AGENT=${TEAM_CODE_ROOT}/mit_agent.py
ENV TEAM_CONFIG=${TEAM_CODE_ROOT}/

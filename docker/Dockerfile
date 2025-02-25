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

# DO NOT EDIT THIS LINE!!!
# This command copies your agent's code into the correct place in the container.
ENV TEAM_CODE_ROOT=/workspace/team_code
COPY team_code ${TEAM_CODE_ROOT}

# Install python library required by your agent.
# RUN python3 -m pip install -r ${TEAM_CODE_ROOT}/requirements.txt
# RUN python3 -m pip install \
#         torch

# Point to your agent's entrypoint file and configuration file
# ONLY MODIFY <AGENT_ENTRYPOINT_FILENAME> and <CONFIG_ENTRYPOINT_FILENAME>
# placeholders. These filenames need to be relative to your agent's team code
# folder.
ENV TEAM_AGENT=${TEAM_CODE_ROOT}/<AGENT_ENTRYPOINT_FILENAME>
ENV TEAM_CONFIG=${TEAM_CODE_ROOT}/<CONFIG_ENTRYPOINT_FILENAME>

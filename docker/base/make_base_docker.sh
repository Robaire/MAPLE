#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DOC_STRING="Build base stack image."

USAGE_STRING=$(cat <<- END
Usage: $0 [-h|--help] [-u|--ubuntu-distro UBUNTU_DISTRO] [-p|--python-version PYTHON_VERSION]

The following UBUNTU distributions are supported:
    * 20.04
    * 22.04

The following PYTHON versions are supported:
    * 3.8
    * 3.9
    * 3.10

Possible combinations:
    * Ubuntu 20.04 - Python 3.8
    * Ubuntu 20.04 - Python 3.9
    * Ubuntu 22.04 - Python 3.10
END
)

usage() { echo "$DOC_STRING"; echo "$USAGE_STRING"; exit 1; }

UBUNTU_DISTRO="22.04"
PYTHON_VERSION="3.10"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -u |--ubuntu-distro )
      UBUNTU_DISTRO=$2
      if [ "$UBUNTU_DISTRO" != "20.04" ] && [ "$UBUNTU_DISTRO" != "22.04" ]; then
        usage
      fi
      shift 2 ;;
    -p |--python-version )
      PYTHON_VERSION=$2
      if [ "$PYTHON_VERSION" != "3.8" ] && [ "$PYTHON_VERSION" != "3.9" ] && [ "$PYTHON_VERSION" != "3.10" ]; then
        usage
      fi
      shift 2 ;;
    -h | --help )
      usage
      ;;
    * )
      shift ;;
  esac
done

rm -fr .lbtmp
mkdir -p ${SCRIPT_DIR}/.lbtmp

CARLA_ROOT=${SCRIPT_DIR}/../../LunarSimulator
LEADERBOARD_ROOT=${SCRIPT_DIR}/../../Leaderboard

echo "Copying CARLA Python API from $CARLA_ROOT"
mkdir -p ${SCRIPT_DIR}/.lbtmp/carla/egghouse ${SCRIPT_DIR}/.lbtmp/carla/wheelhouse
cp -fr ${CARLA_ROOT}/PythonAPI/carla/dist/*.egg  ${SCRIPT_DIR}/.lbtmp/carla/egghouse
cp -fr ${CARLA_ROOT}/PythonAPI/carla/dist/*.whl  ${SCRIPT_DIR}/.lbtmp/carla/wheelhouse

echo "Copying Leaderboard"
mkdir -p ${SCRIPT_DIR}/.lbtmp/leaderboard
cp -fr ${LEADERBOARD_ROOT}/leaderboard ${SCRIPT_DIR}/.lbtmp/leaderboard
cp -fr ${LEADERBOARD_ROOT}/data ${SCRIPT_DIR}/.lbtmp/leaderboard
cp -fr ${LEADERBOARD_ROOT}/requirements.txt ${SCRIPT_DIR}/.lbtmp/leaderboard

echo "Building docker"
docker build \
  --force-rm  \
  --build-arg UBUNTU_DISTRO=$UBUNTU_DISTRO \
  --build-arg PYTHON_VERSION=$PYTHON_VERSION \
  -t lac-leaderboard:ubuntu${UBUNTU_DISTRO}-py${PYTHON_VERSION} \
  -f ${SCRIPT_DIR}/Dockerfile.base ${SCRIPT_DIR}/.lbtmp

rm -fr ${SCRIPT_DIR}/.lbtmp

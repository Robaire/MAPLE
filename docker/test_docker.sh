#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DOC_STRING="Test agent image."

USAGE_STRING=$(cat <<- END
Usage: $0 [-h|--help] [-i|--image IMAGE]

END
)

usage() { echo "$DOC_STRING"; echo "$USAGE_STRING"; exit 1; }

USER_IMAGE="lac-user"
TEST_IMAGE="lac-test"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i |--image )
      USER_IMAGE=$2
      shift 2 ;;
    -h | --help )
      usage
      ;;
    * )
      shift ;;
  esac
done

if [ -z $USER_IMAGE ]; then
  echo "The provided docker image does not exits or is not valid. Please, provide a valid docker image."
  echo ""
  usage
  exit 1
fi

rm -rf ${SCRIPT_DIR}/.lbtmp
mkdir -p ${SCRIPT_DIR}/.lbtmp

echo "Creating testbed for ${USER_IMAGE}..."

cp -fr ${SCRIPT_DIR}/test/run_leaderboard.sh ${SCRIPT_DIR}/.lbtmp/run_leaderboard.sh
cp -fr ${SCRIPT_DIR}/test/entrypoint.sh ${SCRIPT_DIR}/.lbtmp/entrypoint.sh

docker build \
  --quiet \
  --force-rm  \
  --build-arg USER_IMAGE=$USER_IMAGE \
  -t $TEST_IMAGE \
  -f ${SCRIPT_DIR}/test/Dockerfile.test ${SCRIPT_DIR}/.lbtmp

rm -fr ${SCRIPT_DIR}/.lbtmp

docker run \
    -it \
    --rm \
    --network=host \
    --runtime=nvidia \
    --env=NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all} \
    --env=NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}all \
    ${TEST_IMAGE} /bin/bash /workspace/run_leaderboard.sh

echo "Done."
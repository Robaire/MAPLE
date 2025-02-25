#!/bin/bash

DOC_STRING="Build your AGENT docker image."

USAGE_STRING=$(cat <<- END
Usage: $0 -t|--team-code TEAM_CODE_ROOT [-n|--target-name] [-h|--help]

Arguments:
  -t, --team-code         Path to the agent's folder
  -n, --target-name       Docker image target name

END
)

usage() { echo "$DOC_STRING"; echo "$USAGE_STRING"; exit 1; }

TEAM_CODE_ROOT=""
TARGET_NAME="lac-user"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t |--team-code )
      TEAM_CODE_ROOT=$2
      shift 2 ;;
    -n |--target-name )
      TARGET_NAME=$2
      shift 2 ;;
    -h | --help )
      usage
      ;;
    * )
      shift ;;
  esac
done

if [ -z $TEAM_CODE_ROOT ] || ! [ -d $TEAM_CODE_ROOT ]; then
  echo "The provided directory is not valid or does not exist. Please, provide a valida team code root directory."
  echo ""
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rm -rf ${SCRIPT_DIR}/.dtmp
mkdir -p ${SCRIPT_DIR}/.dtmp

echo "Copying team code from $TEAM_CODE_ROOT"
cp -fr ${TEAM_CODE_ROOT} ${SCRIPT_DIR}/.dtmp/team_code

echo "Building docker"
docker build \
  --force-rm  \
  -t ${TARGET_NAME} \
  -f ${SCRIPT_DIR}/Dockerfile ${SCRIPT_DIR}/.dtmp

rm -fr ${SCRIPT_DIR}/.dtmp

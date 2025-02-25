#!/bin/bash
set -e

PYTHON_VERSION=$(python3 -c "import sys; print('{}.{}'.format(sys.version_info[0], sys.version_info[1]))")
export PYTHONPATH=${CARLA_EGGHOUSE_ROOT}/$(ls ${CARLA_EGGHOUSE_ROOT} | grep py${PYTHON_VERSION}):${PYTHONPATH}
export PYTHONPATH=${LEADERBOARD_ROOT}:${PYTHONPATH}

exec "$@"

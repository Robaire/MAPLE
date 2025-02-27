#!/bin/bash

python3 "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
    --missions="/workspace/leaderboard/data/missions_training.xml" \
    --missions-subset="0" \
    --repetitions="1" \
    --checkpoint="/workspace/results" \
    --agent=${TEAM_AGENT} \
    --agent-config=${TEAM_CONFIG} \
    --record="1" \
    --record-control="1" \
    --resume="" \
    --qualifier="" \
    --evaluation="1" \
    --development="1" \
    --testing="1"

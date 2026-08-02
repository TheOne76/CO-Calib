#!/usr/bin/env bash
set -e

source /opt/ros/noetic/setup.bash --extend
source /catkin_ws/devel/.private/kalibr/setup.bash --extend
export KALIBR_MANUAL_FOCAL_LENGTH_INIT=1
export MPLBACKEND=Agg
exec "$@"

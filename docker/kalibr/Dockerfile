FROM ros:noetic-ros-base-focal

ARG KALIBR_BUILD_JOBS=4
ENV DEBIAN_FRONTEND=noninteractive
ENV WORKSPACE=/catkin_ws

RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf automake doxygen git wget \
    python3-catkin-tools python3-dev python3-igraph python3-matplotlib \
    python3-numpy python3-pip python3-pyx python3-scipy python3-tk python3-wxgtk4.0 python3-yaml \
    libblas-dev libboost-all-dev libeigen3-dev liblapack-dev libopencv-dev \
    libpoco-dev libsuitesparse-dev libtbb-dev libv4l-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p ${WORKSPACE}/src && \
    cd ${WORKSPACE} && \
    catkin init && \
    catkin config --extend /opt/ros/noetic --cmake-args -DCMAKE_BUILD_TYPE=Release

COPY vendor/kalibr ${WORKSPACE}/src/kalibr
RUN cd ${WORKSPACE} && catkin build -j${KALIBR_BUILD_JOBS}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-cairo ros-noetic-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

COPY src /opt/omnicalib/src
COPY docker/kalibr/run_calibration.py /usr/local/bin/omnicalib-calibrate
COPY docker/kalibr/extract_detections.py /usr/local/bin/omnicalib-acv-extract
COPY docker/kalibr/entrypoint.sh /usr/local/bin/omnicalib-entrypoint
RUN chmod +x /usr/local/bin/omnicalib-calibrate /usr/local/bin/omnicalib-acv-extract /usr/local/bin/omnicalib-entrypoint

WORKDIR /data
ENTRYPOINT ["/usr/local/bin/omnicalib-entrypoint"]
CMD ["bash"]

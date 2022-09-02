# FROM anibali/pytorch:1.8.1-cuda11.1-ubuntu20.04

From pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

RUN apt-get install -y gcc
RUN pip install pandas
RUN pip install evalutils
RUN pip install meshio
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output
USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip


COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
# Install required python packages via pip - you may adapt the requirements.txt to your needs
RUN python -m pip install --user -rrequirements.txt
# Copy all required files such that they are available within the docker image (code, weights, ...)
RUN pip install matplotlib
RUN pip install -U numpy

COPY --chown=algorithm:algorithm model/ /opt/algorithm/model/
COPY --chown=algorithm:algorithm util/ /opt/algorithm/util/
COPY --chown=algorithm:algorithm ckpt/ /opt/algorithm/ckpt/

# these are only for testing
COPY --chown=algorithm:algorithm dummy/ /opt/algorithm/dummy/
# COPY --chown=algorithm:algorithm test/ /opt/algorithm/test/
# COPY --chown=algorithm:algorithm output/ /opt/algorithm/output/
COPY --chown=algorithm:algorithm input/ /input/

COPY --chown=algorithm:algorithm inference_P2ILF-test_docker.py /opt/algorithm/



# COPY --chown=algorithm:algorithm test/ /opt/algorithm/test/
# COPY --chown=algorithm:algorithm output/ /opt/algorithm/output/
# Entrypoint to your python code - executes process.py as a script
ENTRYPOINT python -m inference_P2ILF-test_docker $0 $@

# you can go in the container and check if you want 
# ENTRYPOINT /bin/bash
## ALGORITHM LABELS ##

# These labels are required
# LABEL nl.diagnijmegen.rse.algorithm.name=P2ILF

# These labels are required and describe what kind of hardware your algorithm requires to run.
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=2
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=16G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.cuda_compute_capability=6.0
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=8G



# sudo docker build -t p2ilf .
# sudo docker run -ti --rm p2ilf:latest /bin/bash
# sudo docker run -ti --rm p2ilfv2:latest /bin/bash
# sudo docker save p2ilf:latest | gzip -c > p2ilf2022.tar.gz
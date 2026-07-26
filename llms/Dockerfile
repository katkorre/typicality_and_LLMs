FROM nvcr.io/nvidia/pytorch:26.05-py3


# ENV Variables
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/include/"

WORKDIR /
COPY /llmprototypes /llmprototypes

ENV NUMBA_CACHE_DIR="/llmprototypes/numba/"
ENV PYTHONPATH="$PYTHONPATH:/llmprototypes"

WORKDIR /llmprototypes
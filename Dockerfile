FROM ubuntu:18.04

RUN apt-get update \
    && apt-get install -y python3-pip python3-dev \
    && cd /usr/local/bin \
    && ln -s /usr/bin/python3 python \
    && pip3 install --upgrade pip

RUN apt-get install ffmpeg libsm6 libxext6 nano -y

WORKDIR /home/transformer_sw

COPY requirements.txt ${PWD}

RUN pip3 install -r requirements.txt

WORKDIR /home/transformer_sw

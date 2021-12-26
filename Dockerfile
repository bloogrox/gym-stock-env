FROM python:3.9

RUN apt-get -y update \
    && apt-get install -y \
    unzip libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb \
    patchelf ffmpeg cmake swig

WORKDIR /usr/local/gym/

ENV PYTHONPATH /usr/local/gym/

COPY . /usr/local/gym/

RUN pip install -U pip

RUN pip install -r requirements.txt

ENTRYPOINT ["/usr/local/gym/bin/docker_entrypoint"]

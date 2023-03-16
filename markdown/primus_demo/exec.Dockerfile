FROM ubuntu:22.04

# Install development tools
RUN apt update
RUN apt install -yq vim tree iputils-ping net-tools telnet wget curl dnsutils unzip

# build tools
RUN apt install -yq build-essential git gcc-9 g++-9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 40 --slave /usr/bin/g++ g++ /usr/bin/g++-9 --slave /usr/bin/gcov gcov /usr/bin/gcov-9

# python and its dependencies
RUN apt install -yq libfuse-dev zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev libhdf5-dev libbz2-dev
RUN curl -O https://www.python.org/ftp/python/3.8.6/Python-3.8.6.tar.xz && \
  tar -xf Python-3.8.6.tar.xz && \
  cd Python-3.8.6 && \
  ./configure --enable-optimizations && \
  make -j"$(nproc)" build_all && \
  make altinstall && \
  cd .. && rm -rf Python-3.8.6.tar.xz && rm -rf Python-3.8.6
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.8 3
RUN update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.8 3
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN wget https://bootstrap.pypa.io/get-pip.py && \
  python3.8 get-pip.py && \
  python3.8 -m pip install -U pip && \
  python3.8 -m pip freeze | xargs python3.8 -m pip uninstall -y || true && \
  python3.8 -m pip install --no-cache-dir numpy==1.19.4 nltk==3.6.7 scipy google-cloud-storage==1.35.0

RUN pip3.8 install --no-cache-dir kafka-python tensorflow==2.4.0
# installing specific protobuf version solves "TypeError: Descriptors cannot not be created directly."
RUN pip3.8 install --no-cache-dir tensorflow-datasets protobuf==3.20.*
ADD ./monolith-0.1.0-cp38-cp38-linux_x86_64.whl /root/
RUN pip3.8 install --no-cache-dir /root/monolith-0.1.0-cp38-cp38-linux_x86_64.whl

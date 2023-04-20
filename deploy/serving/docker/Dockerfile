FROM debian:buster-20221219

LABEL maintainer="Monolith"

ARG PYPI_SOURCE=https://pypi.tuna.tsinghua.edu.cn/simple

# pre install for tsinghua apt source
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        apt-transport-https ca-certificates wget dirmngr gnupg \
        software-properties-common \
    ;

# install java
RUN set -eux; \
    wget -qO - https://adoptopenjdk.jfrog.io/adoptopenjdk/api/gpg/key/public | apt-key add - ; \
    add-apt-repository --yes https://adoptopenjdk.jfrog.io/adoptopenjdk/deb/; \
    apt-get update; \
    apt-get install -y --no-install-recommends adoptopenjdk-8-hotspot

ENV JAVA_HOME /usr/lib/jvm/adoptopenjdk-8-hotspot-amd64/
ENV JRE_HOME ${JAVA_HOME}/jre
ENV CLASSPATH .:${JAVA_HOME}/lib:${JRE_HOME}/lib
ENV PATH ${JAVA_HOME}/bin:$PATH

# copy assets
ADD deploy/serving/docker/assets /tmp/assets
# Copy the service of dumping environment variables
COPY deploy/serving/docker/assets/configurator_dumpenv.sh /root/.system/
COPY deploy/serving/docker/assets/configurator_dumpenv.service /etc/systemd/system/
# Configure bashrc
COPY deploy/serving/docker/assets/bashrc /root/.bashrc

# Change mirrors in /etc/apt/sources.list to tsinghua mirrors
RUN set -eux; \
    { \
        echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ buster main contrib non-free"; \
        echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ buster-updates main contrib non-free"; \
        echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ buster-backports main contrib non-free"; \
        echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian-security buster/updates main contrib non-free"; \
    } > /etc/apt/sources.list; \
    \
# Install required packages
    apt-get update; \
    apt-get upgrade -y; \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        curl lsof procps locales tzdata less vim python \
        build-essential autoconf automake bzip2 file imagemagick libbz2-dev \
        libc6-dev openssl libcurl4-openssl-dev libdb-dev libevent-dev \
        libffi-dev libgdbm-dev libgeoip-dev libglib2.0-dev libjpeg-dev \
        libkrb5-dev liblzma-dev libmagickcore-dev libmagickwand-dev \
        libncurses-dev libpng-dev libpq-dev libreadline-dev libsqlite3-dev \
        default-libmysqlclient-dev libssl-dev libtool libwebp-dev libxml2 \
        libxml2-dev libxslt-dev libyaml-dev make patch xz-utils zlib1g-dev \
        tcl tk git rsync ssh net-tools iputils-ping pbzip2 python-dev netcat \
        libxslt1-dev libcap2-bin libjemalloc2 libjemalloc-dev libsnappy1v5 \
        libtcmalloc-minimal4 libzookeeper-mt2 lldpd libnss3 pv gnupg2 libaio1 \
        systemd systemd-sysv libsystemd0 netcat-openbsd rsyslog unscd \
        apt-utils cgroup-bin cmake gdb libncurses5-dev libnss3-dev \
        libprotobuf-dev linux-base linux-libc-dev linux-perf \
        openssh-client openssh-server protobuf-compiler  \
        python-pip python-pycurl python-setuptools python-wheel python3-dev \
        python3-pip python3-pycurl python3-setuptools python3-wheel sysstat \
        telnet tree libunwind-dev numactl unzip \
    ; \
# Remove apt cache
    rm -rf /var/lib/apt/lists/*

# RDMA Essentials
# download from https://linux.mellanox.com/public/repo/mlnx_rdma_minimal/5.0-1.0.0.0/debian10.0/
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends \
    libnl-3-200=3.4.0-1 libnl-3-dev libnl-route-3-200=3.4.0-1 libnl-route-3-dev; \
    dpkg -i /tmp/assets/rdma/librdmacm1_50mlnx1-1.50100.0_amd64.deb \
        /tmp/assets/rdma/rdmacm-utils_50mlnx1-1.50100.0_amd64.deb \
        /tmp/assets/rdma/libibverbs1_50mlnx1-1.50100.0_amd64.deb \
        /tmp/assets/rdma/ibverbs-utils_50mlnx1-1.50100.0_amd64.deb \
        /tmp/assets/rdma/libibverbs-dev_50mlnx1-1.50100.0_amd64.deb \
        /tmp/assets/rdma/ibverbs-providers_50mlnx1-1.50100.0_amd64.deb \
        /tmp/assets/rdma/libibumad3_50mlnx1-1.50100.0_amd64.deb

RUN set -eux; \
# Create tiger account
    if ! id -u tiger >/dev/null 2>&1 ; then \
        groupadd -f tiger; \
        useradd -u 1000 -g tiger -d /home/tiger -m -s /bin/bash tiger; \
    fi; \
    mkdir -p /home/tiger/.service/ /opt/tiger /opt/log/tiger /var/log/tiger; \
    chown tiger:tiger /home/tiger/.service/ /opt/tiger /opt/log/tiger /var/log/tiger; \
# Change timezone
    echo "Asia/Shanghai" > /etc/timezone; \
    ln -sfn /usr/share/zoneinfo/Asia/Shanghai /etc/localtime; \
    \
# Generate locales
    sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen; \
    sed -i 's/# zh_CN.UTF-8 UTF-8/zh_CN.UTF-8 UTF-8/' /etc/locale.gen; \
    locale-gen; \
# Configure permissions
    chmod 700 /root/.system/configurator_dumpenv.sh; \
# Install pip
    tar -xf /tmp/assets/Python-3.8.6.tar.xz && cd Python-3.8.6 && \
    ./configure --enable-optimizations && make -j8 build_all && make altinstall && \
    update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.8 3 && \
    ln -s /usr/share/pyshared/lsb_release.py /usr/local/lib/python3.8/site-packages/lsb_release.py && \
    cd .. && rm -rf Python-3.8.* && \
    cd /usr/local/bin && ln -s pip3.8 pip3 && \

    cp /tmp/assets/pip.conf /etc/ && mkdir ~/.pip && cp /tmp/assets/pip.conf ~/.pip/ && \
    printf "\n. /etc/profile\n" >> /root/.bashrc && printf "\n. /etc/profile\n" >> /home/tiger/.bashrc && \
    sh /tmp/assets/build.sh && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache/pip && \
# Enable the service of dumping environment variables
    systemctl enable configurator_dumpenv.service; \
# Configure systemd
    sed -i '/#DefaultLimitNOFILE=/c\DefaultLimitNOFILE=1048576:1048576' /etc/systemd/system.conf; \
    rm -fr /lib/systemd/system/multi-user.target.wants/* \
        /etc/systemd/system/*.wants/* \
        /lib/systemd/system/local-fs.target.wants/* \
        /lib/systemd/system/sockets.target.wants/*udev* \
        /lib/systemd/system/sockets.target.wants/*initctl* \
        /lib/systemd/system/sysinit.target.wants/systemd-tmpfiles-setup* \
        /lib/systemd/system/systemd-update-utmp* \
        /etc/systemd/system/-.mount \
        /lib/systemd/system/user-.slice.d/; \
# Remove cron.daily
    rm -f /etc/cron.daily/*

# Set locales to en_US.UTF-8
ENV LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 LC_CTYPE=en_US.UTF-8
ENV IS_DOCKER_ENV=true

# Set the default mount path, the path /sys/fs/cgroup is necessary to systemd;
VOLUME ["/sys/fs/cgroup","/run","/run/lock","/tmp"]

# Currently the service is managed by systemd. So systemd needs to be installed and set as the default command.
CMD ["/lib/systemd/systemd"]

# CUDA Essentials
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
        gawk \
        binutils-dev \
        && \
    curl -fsSL http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub | apt-key add - && \
    echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    curl -fsSL http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    apt-get install -yq --no-install-recommends --fix-missing \
        cuda-cudart-11-0=11.0.221-1 \
        cuda-compat-11-0 \
        cuda-libraries-11-0=11.0.3-1 \
        libnpp-11-0=11.1.0.245-1 \
        cuda-nvtx-11-0=11.0.167-1 \
        libcublas-11-0=11.2.0.252-1 \
        cuda-nvml-dev-11-0=11.0.167-1 \
        cuda-command-line-tools-11-0=11.0.3-1 \
        cuda-nvprof-11-0=11.0.221-1 \
        libnpp-dev-11-0=11.1.0.245-1 \
        cuda-libraries-dev-11-0=11.0.3-1 \
        cuda-minimal-build-11-0=11.0.3-1 \
        libcublas-dev-11-0=11.2.0.252-1 \
        libcusparse-11-0=11.1.1.245-1 \
        libcusparse-dev-11-0=11.1.1.245-1 \
        libcudnn8=8.0.5.39-1+cuda11.0 \
        libcudnn8-dev=8.0.5.39-1+cuda11.0 \
        libnccl2=2.8.3-1+cuda11.0 \
        libnccl-dev=2.8.3-1+cuda11.0 \
        libnvinfer7=7.2.1-1+cuda11.0 \
        libnvinfer-dev=7.2.1-1+cuda11.0 \
        libnvinfer-plugin7=7.2.1-1+cuda11.0 \
        libnvinfer-plugin-dev=7.2.1-1+cuda11.0 \
        libxml-sax-expat-perl libexpat1 libexpat1-dev \
        && \
    ln -s /usr/local/cuda-11.0 /usr/local/cuda && \
    find /usr/local/cuda-11.0/lib64/ -type f -name '*.a' -not -name 'libcudart_static.a' -not -name 'libcudadevrt.a' -delete && \
    rm /etc/alternatives/libcudnn_stlib && \
    rm /usr/lib/x86_64-linux-gnu/libcudnn_static.a && \
    rm /usr/lib/x86_64-linux-gnu/libcudnn_static_v8.a && \
    rm /usr/lib/x86_64-linux-gnu/libnccl_static.a && \
    rm /usr/lib/x86_64-linux-gnu/libnvinfer_static.a && \
    rm /usr/lib/x86_64-linux-gnu/libnvinfer_plugin_static.a && \
    rm /usr/lib/x86_64-linux-gnu/libmyelin_compiler_static.a && \
    rm /usr/lib/x86_64-linux-gnu/libmyelin_executor_static.a && \
    rm /usr/lib/x86_64-linux-gnu/libmyelin_pattern_library_static.a && \
    rm /usr/lib/x86_64-linux-gnu/libmyelin_pattern_runtime_static.a && \
    rm /etc/apt/sources.list.d/cuda.list && \
    rm /etc/apt/sources.list.d/nvidia-ml.list

RUN python3.8 -m pip install ifstat==1.0.3 absl-py==0.12.0 kazoo==2.8.0 Flask-API==2.0 \
    dataclasses-json==0.5.2 numpy==1.23.4 psutil==5.8.0 msgpack==1.0.2 \
    pyinotify==0.9.6 Jinja2==2.11.3 requests==2.25.1 PyYAML==3.13 redis==3.5.1 \
    protobuf==3.12.4 grpcio==1.26.0 sqlalchemy==1.3.24 tensorflow-gpu==2.4.0
RUN HOROVOD_NCCL_LINK=SHARED HOROVOD_WITHOUT_GLOO=1 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_GPU_OPERATIONS=NCCL python3.8 -m pip install --no-cache-dir horovod==0.21.3

# CUDA environment variables
ENV CUDA_HOME "/usr/local/cuda"
ENV PATH "${CUDA_HOME}/bin:${PATH}"
# ENV LD_LIBRARY_PATH "${CUDA_HOME}/compat:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH "${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.0 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 brand=tesla,driver>=450,driver<451"

# hadoop
RUN wget -q https://mirrors.aliyun.com/apache/hadoop/common/hadoop-3.3.2/hadoop-3.3.2.tar.gz && \
    tar -xzf hadoop-3.3.2.tar.gz && mv hadoop-3.3.2 /opt/tiger/hadoop && rm hadoop-3.3.2.tar.gz

ENV HADOOP_HOME /opt/tiger/hadoop/
ENV HDFS_JDK ${JAVA_HOME}
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${HADOOP_HOME}/lib/native:${JAVA_HOME}/jre/lib/amd64/server
ENV CLASSPATH ${CLASSPATH}:`${HADOOP_HOME}/bin/hadoop classpath --glob`
ENV HADOOP_HDFS_HOME $HADOOP_HOME
ENV HADOOP_OPTS "-Djava.library.path=$HADOOP_HOME/lib/native"
ENV PATH $PATH:$HADOOP_HOME/sbin:$HADOOP_HOME/bin

RUN printf "\nexport PATH=${JAVA_HOME}/bin:${CUDA_HOME}/bin:$HADOOP_HOME/sbin:$HADOOP_HOME/bin:$PATH\n" >> /root/.bashrc && \
    printf "\nexport PATH=${JAVA_HOME}/bin:${CUDA_HOME}/bin:$HADOOP_HOME/sbin:$HADOOP_HOME/bin:$PATH\n" >> /home/tiger/.bashrc


# ADD monolith_serving /opt/tiger/monolith_serving

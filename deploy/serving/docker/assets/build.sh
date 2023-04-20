#!/usr/bin/env bash
set -ex

# still make python2 as default, but python 3.8 is installed

export PYTHON_PIP_VERSION=20.1
curl -skSLf -o get-pip.py 'https://bootstrap.pypa.io/pip/2.7/get-pip.py'

python get-pip.py \
    --disable-pip-version-check \
    --no-cache-dir \
    -i http://mirrors.aliyun.com/pypi/simple \
    --trusted-host mirrors.aliyun.com \
    "pip==$PYTHON_PIP_VERSION"

find /usr/local -depth \
    \( \
        \( -type d -a -name test -o -name tests \) \
        -o \
        \( -type f -a -name '*.pyc' -o -name '*.pyo' \) \
    \) -exec rm -rf '{}' +;
rm -f get-pip.py
rm -rf /usr/src/python

cd /root/

pip uninstall -y paramiko pycrypto
pip install --no-cache-dir setuptools==44.1.0 virtualenv==15.1.0 cffi==1.12.3 paramiko==1.18.3
pip install --no-cache-dir -r /tmp/assets/requirements.txt

# systemd
[ -d /etc/systemd/system/user@1000.service.d ] || mkdir /etc/systemd/system/user@1000.service.d
echo "[Service]
Restart=always" > /etc/systemd/system/user@1000.service.d/always.conf
echo "[Service]
LimitNOFILE=1000000
LimitMEMLOCK=infinity" > /etc/systemd/system/user@1000.service.d/limits.conf

## for run systemd
cd /lib/systemd/system/sysinit.target.wants/ && \
    ls | grep -v systemd-tmpfiles-setup | xargs rm -f $1 && \
    rm -f /lib/systemd/system/sockets.target.wants/*udev*

systemctl mask -- \
    apt-daily-upgrade.timer \
    apt-daily.timer \
    cgmanager.service \
    cgproxy.service \
    dev-mqueue.mount \
    getty-static.service \
    getty.target \
    swap.target \
    systemd-logind.service \
    systemd-remount-fs.service \
    systemd-timesyncd.service \
    systemd-tmpfiles-setup-dev.service \
    systemd-tmpfiles-setup.service \
    systemd-update-utmp-runlevel.service; \
    tmp.mount \
    etc-hostname.mount \
    etc-hosts.mount \
    etc-resolv.conf.mount \
    -.mount \

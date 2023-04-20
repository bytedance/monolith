#!/bin/bash - 

xargs -0 bash -c 'printf "%q\n" "$@" ; systemctl set-environment "$@"' -- \
    < /proc/1/environ \
    > /var/docker_environment

chmod 700 /var/docker_environment

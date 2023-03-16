#!/bin/bash

set -ex

# setup env
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$JAVA_HOME/jre/lib/amd64/server/
export HADOOP_HDFS_HOME=/usr/lib/hadoop
export CLASSPATH="$(/usr/lib/hadoop/bin/hadoop classpath --glob)"

python3 demo/demo_model.py \
--model_dir=hdfs:///primus/model-checkpoints/movie_lens_tutorial \
--model_name=movie_lens_tutorial \
--training_type=stdin
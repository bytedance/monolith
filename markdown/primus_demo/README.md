# Monolith x Primus Demo

## Setup Primus

Follow the primus quickstart guide to setup the primus baseline virtual machine: https://github.com/bytedance/primus/blob/master/docs/primus-quickstart.md

Then, replace /usr/lib/primus-kubernetes/conf/default.conf with https://code.byted.org/inf/primus/blob/hopang/baseline/feature/tensorflow-examples/deployments/baseline-kubernetes/conf/default.conf

## Setup Monolith

In your virtual machine, clone the open source monolith

```bash
cd
git clone https://github.com/bytedance/monolith
```


```bash
vim monolith/tools/pip_package/BUILD
vim monolith/tools/pip_package/build_pip_package.sh
```

After editing is finished, copy the `tools` and `markdown` directory to your virtual machine

```
scp -i xxx.pem -r monolith/tools ubuntu@VM_IP:/home/ubuntu/monolith/
scp -i xxx.pem -r monolith/markdown ubuntu@VM_IP:/home/ubuntu/monolith/
```

#### Prepare environment and compile in your virtual machine

Monolith can be compiled on ubuntu 22.04 by first building the compilation docker: [compile.Dockerfile](./compile.Dockerfile). In your virtual machine,

build the compile docker
```bash
cd monolith/markdown/primus_demo
docker build -t monolith_ubuntu22:1.0 -f compile.Dockerfile .
```

run the compile docker and bind mount monolith
```bash
cd
export COMPILE_DOCKER_ID=`docker run -itd -v $(pwd)/monolith:/monolith monolith_ubuntu22:1.0 bash`
docker exec -it $COMPILE_DOCKER_ID bash
```

Then, in the docker, build monolith
```bash
bazel build //monolith/native_training:cpu_training
```

Verify the build by running the demo. Note that you don't need to actually finish running the demo. If you see a progressbar, the build should be ok and you can kill the demo. 
```
pip3 install tensorflow_datasets
bazel run //markdown/demo:demo_local_runner
```

Finally, package files to `.whl`
```
bazel run //monolith/tools/pip_package:build_pip_package
```

When build is finished, the `.whl` file should be located in `~/monolith/pip_monolith`. 

### Prepare monolith execution image

```bash
cd monolith/markdown/primus_demo
# copy monolith wheel to the current directory
docker cp $COMPILE_DOCKER_ID:/root/pip_monolith/monolith-0.1.0-cp38-cp38-linux_x86_64.whl .
docker build -t monolith_ubuntu22_exec:1.0 -f exec.Dockerfile .
```

Then, load this image to k8s cluster

```bash
kind load docker-image monolith_ubuntu22_exec:1.0
```

[//]: # (END_INTERNAL)

### Prepare monolith image

```bash
cd monolith/markdown/primus_demo
docker build -t monolith_ubuntu22_exec:1.0 -f monolith.Dockerfile .
```

Then, load this image to k8s cluster

```bash
kind load docker-image monolith_ubuntu22_exec:1.0
```

## Prepare the movie-lens dataset

Now, we will convert the movie-lens dataset to CSV format, which is later feed to the model through Primus's input manager. This may take a while (a few hours) due to the size of the dataset, depending on the number of CPU cores you have. 

```bash
pip3 install tensorflow==2.4.0 tensorflow-datasets
cd monolith/markdown/demo
mkdir -p data_1m
python3 ml_dataset.py
```

When the conversion finished, upload the data to HDFS

```bash
/usr/lib/hadoop/bin/hdfs dfs -put data_1m /primus/
```

## Launch training with Primus on k8s

First, make sure that the `files` entry of `monolith/markdown/primus_demo/primus_monolith.json` matches the actual place where you clone monolith. Then, you can submit the training via

```bash
/usr/lib/primus-kubernetes/sbin/primus-submit --primus_conf primus_monolith.json
```

[//]: # (BEGIN_INTERNAL)

## TODOs

1. Parsing from stdin using Python has serious performance issue. In house training utilizes special PBDataset kernels to handle this case, but it's not suited for the open source use case.

[//]: # (END_INTERNAL)
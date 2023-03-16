# Monolith x Primus Demo

## Setup Primus

Follow the primus quickstart guide to setup the primus baseline virtual machine: https://github.com/bytedance/primus/blob/master/docs/primus-quickstart.md

## Setup Monolith

In your virtual machine, clone the open source monolith

```bash
cd
git clone https://github.com/bytedance/monolith
```


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


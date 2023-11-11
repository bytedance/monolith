
 Monolith

## What is it?

[Monolith](https://arxiv.org/abs/2209.07663) is a deep learning framework for large scale recommendation modeling. It introduces two important features which are crucial for advanced recommendation system: 
* collision-less embedding tables guarantees unique representation for different id features
* real time training captures the latest hotspots and help users to discover new interests rapidly

Monolith is built on the top of Tensorflow and supports batch/real-time training and serving.


## Discussion Group

### Join us at Discord

https://discord.gg/QYTDeKxGMX

## Quick start

### Build from source

Currently, we only support compilation on the Linux.

First, download bazel 3.1.0
```bash
wget https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel-3.1.0-installer-linux-x86_64.sh && \
  chmod +x bazel-3.1.0-installer-linux-x86_64.sh && \
  ./bazel-3.1.0-installer-linux-x86_64.sh && \
  rm bazel-3.1.0-installer-linux-x86_64.sh
```

Then, prepare a python environment
```bash
pip install -U --user pip numpy wheel packaging requests opt_einsum
pip install -U --user keras_preprocessing --no-deps
```

Finally, you can build any target in the monolith.
For example,
```bash
bazel run //monolith/native_training:demo --output_filter=IGNORE_LOGS
```

### Demo and tutorials

There are a tutorial in [markdown/demo](markdown/demo) on how to run distributed async training, and few guides on how to use the `MonolithModel` API [here](markdown).  
# Monolith

Monolith is a cutting-edge deep learning framework designed specifically for large-scale recommendation systems. Leveraging TensorFlow as its foundation, Monolith introduces innovative features to improve model accuracy and efficiency in real-time recommendation scenarios.

## Key Features

- **Collisionless Embedding Tables**: Ensures unique representations for different ID features, eliminating collisions and improving recommendation accuracy.
- **Real-Time Training**: Captures the latest trends and hotspots, allowing users to discover new interests and enhancing system responsiveness.
- **Scalable Training and Serving**: Supports both batch and real-time training, making it suitable for various large-scale recommendation use cases.

## Why Use Monolith?

Monolith is designed to address the challenges of modern recommendation systems, including:
- Handling billions of unique user and item features.
- Keeping models updated with real-time user behaviors.
- Supporting high-throughput, low-latency prediction.

Ideal for applications such as e-commerce, video streaming, and social media platforms.

---

## Getting Started

### Installation

Monolith currently supports Linux-based systems. Below are the steps to get started.

#### 1. Install Bazel 3.1.0

```bash
wget https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel-3.1.0-installer-linux-x86_64.sh && \
  chmod +x bazel-3.1.0-installer-linux-x86_64.sh && \
  ./bazel-3.1.0-installer-linux-x86_64.sh && \
  rm bazel-3.1.0-installer-linux-x86_64.sh
```

#### 2. Prepare Python Environment

Set up a Python environment and install the required dependencies:

```bash
pip install -U --user pip numpy wheel packaging requests opt_einsum
pip install -U --user keras_preprocessing --no-deps
```

#### 3. Build Monolith

You can now build and run Monolith targets. For example:

```bash
bazel run //monolith/native_training:demo --output_filter=IGNORE_LOGS
```

---

## Tutorials and Documentation

Explore our tutorials and guides to understand how to use Monolith effectively:

- [Distributed Asynchronous Training Tutorial](markdown/demo)
- [MonolithModel API Guides](markdown)

---

## Community and Support

### Join the Discussion

Connect with other developers and contributors:
- **Discord**: [Join our community](https://discord.gg/QYTDeKxGMX)

### Reporting Issues

If you encounter any issues, please open a ticket in the [GitHub Issues](https://github.com/bytedance/monolith/issues) section.

---

## Contributing

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

---

## License

Monolith is licensed under the [Apache 2.0 License](LICENSE). Feel free to use it in your projects while adhering to the license terms.

---

## Badges

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

---

Thank you for using Monolith! Together, let’s build the future of recommendation systems.

# Contributing to Monolith

Thank you for considering contributing to Monolith! Your contributions help improve this lightweight recommendation system. To ensure a smooth collaboration, please follow these guidelines.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Submitting Pull Requests](#submitting-pull-requests)
- [Development Setup](#development-setup)
- [Style Guides](#style-guides)
  - [Coding Standards](#coding-standards)
  - [Commit Messages](#commit-messages)
- [Community](#community)
- [License](#license)

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](https://www.contributor-covenant.org/version/2/0/code_of_conduct/). By participating in this project, you agree to abide by its terms.

## How Can I Contribute?

### Reporting Bugs

If you encounter a bug, please [open an issue](https://github.com/bytedance/monolith/issues) and include:

- A clear and descriptive title.
- Steps to reproduce the issue.
- Expected and actual behavior.
- Any relevant logs or screenshots.

### Suggesting Enhancements

To suggest new features or enhancements, please [open an issue](https://github.com/bytedance/monolith/issues) and provide:

- A clear and descriptive title.
- A detailed description of the proposed enhancement.
- The motivation and use case for the enhancement.

### Submitting Pull Requests

Before submitting a pull request:

1. Fork the repository and create your branch from `master`.
2. Ensure your code follows the project's coding standards.
3. Include appropriate tests for your changes.
4. Update the documentation as needed.
5. Verify that all tests pass.

## Development Setup

To set up the development environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/bytedance/monolith.git
   ```
2. Install [Bazel 3.1.0](https://github.com/bazelbuild/bazel/releases/tag/3.1.0) as the build tool.
3. Build the project:
   ```bash
   bazel build //...
   ```

## Style Guides

### Coding Standards

- Follow the existing code style in the repository.
- Write clear and concise code with appropriate comments.

### Commit Messages

- Use the present tense ("Add feature" not "Added feature").
- Keep messages brief and descriptive.
- Reference issues and pull requests when applicable.

## Community

Join our community discussions on [Discord](https://discord.gg/QYTDeKxGMX) to connect with other contributors and maintainers.

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](https://github.com/bytedance/monolith/blob/master/LICENSE).

---

Thank you for your contributions! 
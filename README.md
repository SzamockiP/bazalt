# Bazalt

Bazalt is a Python library for rapid prototyping and building graphical applications using the Vulkan API.

## Requirements
- Python 3.10+
- CMake (3.28+)
- A C++ compiler supporting the C++23 standard (e.g. MSVC, GCC, Clang)
- Vulkan SDK

## Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

2. Install the library in developer mode using `pip`:
   ```bash
   pip install -e .
   ```

## Running examples

After successful installation, navigate to the `examples/` folder and run one of the provided examples, e.g.:

```bash
cd examples
python test_cube.py
```

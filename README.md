
---

# micrograd++

<img src="/public/german_shephard.jpg" alt="drawing" width="200" height="200"/>

Welcome to micrograd++. This repository is inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). micrograd++ is a pure C++ machine learning library designed to make machine learning accessible to everyone.

## Overview

micrograd++ aims to provide a simple yet powerful framework for building and training machine learning models. By leveraging C++, it ensures performance efficiency and allows for deep integration with C++-based systems and applications.

![Multi layer perceptron](/public/mlp.gif)

## Features

- **Pure C++**: Entirely implemented in C++ for high performance.
- **Inspired by micrograd**: Brings the simplicity and educational value of micrograd to the C++ ecosystem.
- **Accessible Machine Learning**: Designed to be easy to use, even for those new to machine learning or C++.

## Getting Started

### Prerequisites

- **CMake**: Version 3.15 or higher
- **C++ Compiler**: Supports C++17 standard
- **opencv**: For visualization

### Building the Library

1. Clone the repository:
   ```sh
   git clone https://github.com/gautam-sharma1/microgradpp.git
   cd microgradpp
   ```

2. Create a build directory:
   ```sh
   mkdir build
   cd build
   ```

3. Configure the project with CMake:
   ### Build microgradpp only
      ```sh
      cmake ..
      ```
   or
   ### Builds example and tests
   ```sh
   cmake .. -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON ..
   ```
   or
   ### To build a Release build
   ```sh
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```

4. Build the project:
   ```sh
   make
   ```

### Using the Header only Library

Microgradpp is also available as a header only library. 


### Running the Example

After building, you can run the provided example by executing the `m++` binary:
```sh
./m++
```

## Project Structure

- `CMakeLists.txt`: Build configuration for CMake.
- `include/`: Header files for the library.
- `src/`: Source files for the library.
- `main.cpp`: Example usage of the library.

## Usage

Here’s a brief example of how to use micrograd++:

```cpp
#include <iostream>
#include "Value.h"

int main() {
    auto a = microgradpp::Value::create(2.0);
    auto b = microgradpp::Value::create(3.0);
    auto c = a * b;
    c->backProp();
    
    std::cout << "a->grad: " << a->grad << std::endl;
    std::cout << "b->grad: " << b->grad << std::endl;

    return 0;
}
```

## Examples

Head over to the examples directory to play around with the examples. To build examples, configure cmake from the root directory as follows:

```sh
cd build
cmake -DBUILD_EXAMPLES=ON .. && make
cd examples
./example_mlp # to run mlp example
```
This will produce executables in the build folder

### ML example
A simple multi layer perceptron is defined in 
**mlp.cpp**. Running it for 50 iterations gives the following result:

![Loss function of MLP](/public/mlp.png)

### Computer Vision example

The example in **images.cpp** learns a cute german shephard puppy face. The output of it is as follows:

![Neural network predicting a german shephard face](/public/gsd.gif)

## Contributing

We welcome contributions to micrograd++. Here are a few areas where you can help:

### TODO

- ~~**Modify CMakeLists to Add a Flag to Build Tests**: Enhance the build configuration to optionally include tests.~~
- ~~**Make a Tensor Class**: Create a Tensor class to simplify data loading and manipulation.~~
- ~~**Add an Activation Function Enum or Class**: Implement a flexible way to handle different activation functions.~~
- ~~**Make an Abstract Base Class for Layer and Value**: Design abstract base classes to improve the architecture and extensibility.~~
- **CI/CD pipeline**: Develop a pipeline using github actions to execute tests automatically on a commit.
- **Improve README**: Add few examples on how to leverage this library.
- **Python Interface??**: Probably make a python interface  

## License

This project is licensed under the GNU License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Andrej Karpathy**: For the original [micrograd](https://github.com/karpathy/micrograd) library and inspiration.


## Author
- **Gautam Sharma**: [gsharma](https://www.gsharma.dev)

---

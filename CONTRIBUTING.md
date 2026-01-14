# Contributing to microgradpp

First off, thank you for considering contributing to microgradpp! It's people like you that make microgradpp a great educational tool.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Style Guidelines](#style-guidelines)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project and everyone participating in it is governed by our commitment to creating a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.10+ (for building examples and tests)
- Git

### Setting Up Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/microgradpp.git
   cd microgradpp
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/ggsharma/microgradpp.git
   ```
4. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Building and Testing

```bash
mkdir build && cd build
cmake ..
make
ctest --output-on-failure
```

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title** describing the issue
- **Steps to reproduce** the behavior
- **Expected behavior** vs actual behavior
- **Code samples** if applicable
- **Environment details** (compiler, OS, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. Include:

- **Clear title** describing the enhancement
- **Detailed description** of the proposed functionality
- **Use cases** explaining why this would be useful
- **Possible implementation** if you have ideas

### Your First Code Contribution

Look for issues labeled:
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `documentation` - Documentation improvements

### Pull Requests

1. Follow the [style guidelines](#style-guidelines)
2. Update documentation if needed
3. Add tests for new functionality
4. Ensure all tests pass
5. Update the README if adding new features

## Style Guidelines

### C++ Style

```cpp
// Use descriptive names
class ValueNode;  // Good
class VN;         // Bad

// Use smart pointers
std::shared_ptr<Value> value;  // Good
Value* value;                   // Avoid raw pointers

// Modern C++ features
auto result = compute();        // OK when type is clear
ValuePtr result = compute();    // Preferred for clarity

// Const correctness
const std::vector<ValuePtr>& parameters() const;

// Namespace
namespace microgradpp {
    // All code here
}
```

### File Organization

```
include/
  microgradpp/
    Value.hpp       # Core Value class
    Neuron.hpp      # Neural network building blocks
    Layer.hpp
    MLP.hpp
examples/
  basic_example.cpp
tests/
  test_value.cpp
```

### Documentation

- Use Doxygen-style comments for public APIs
- Include examples in documentation
- Keep README up to date

```cpp
/**
 * @brief Computes gradients via backpropagation
 * 
 * Traverses the computational graph in reverse topological order,
 * computing gradients using the chain rule.
 * 
 * @example
 *   auto x = Value::create(2.0);
 *   auto y = x->pow(2);
 *   y->backward();
 *   std::cout << x->grad;  // 4.0
 */
void backward();
```

## Pull Request Process

1. **Create PR** against the `main` branch
2. **Fill out the PR template** completely
3. **Wait for review** - maintainers will review within a few days
4. **Address feedback** - make requested changes
5. **Merge** - once approved, your PR will be merged

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No new warnings

## Recognition

Contributors will be recognized in:
- The README contributors section
- Release notes for significant contributions

Thank you for contributing to microgradpp! 🎉

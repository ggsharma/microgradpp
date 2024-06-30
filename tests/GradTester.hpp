#pragma once

#include <string>
#include <iostream>

namespace microgradpp{
    class GradTester{
    public:
        GradTester() = default;

        template<class T>
        static bool equals(T actual, T expected, std::string testName, std::string diagnostic = "") {
            // Helper function to round to 3 decimal places
            auto roundToThreeDecimals = [](T value) -> T {
                return std::round(value * 1000.0) / 1000.0;
            };

            T roundedActual = roundToThreeDecimals(actual);
            T roundedExpected = roundToThreeDecimals(expected);

            if (roundedActual == roundedExpected) {
                std::cout << "\033[1;32m" << testName << ": PASSED" << "\033[0m" << std::endl; // Green color for PASSED
                return true;
            } else {
                std::cout << "\033[1;31m" << testName << ": FAILED (Expected: " << expected << ", Actual: " << actual << ")" << "\033[0m" << std::endl; // Red color for FAILED
                return false;
            }
        }

    };
}


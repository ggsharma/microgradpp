# pragma once

#include <vector>

#include "Value.hpp"


namespace microgradpp::core{
    // No-op Abstract
    class MppCore{
    public:
        MppCore() = default;
        virtual void print() const = 0;
        virtual void zeroGrad() {};
        virtual void printParameters() {};
        virtual std::vector<ValuePtr> operator()(const std::vector<ValuePtr>& in) = 0;
    };
}

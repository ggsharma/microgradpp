# pragma once

#include <vector>

#include "Value.hpp"
#include "TypeDefs.hpp"


namespace microgradpp::core{
    // No-op Abstract
    class MppCore{
    public:
        MppCore() = default;
        virtual void print() const = 0;
        virtual Tensor1D operator()(const Tensor1D& in) = 0;
        virtual void zeroGrad() {};
        virtual void printParameters() const{};
        __MICROGRADPP_NO_DISCARD__ virtual std::vector<Value*> parameters() const { return {};};
    };
}

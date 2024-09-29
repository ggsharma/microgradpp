#pragma once
#include "Value.hpp"

namespace microgradpp{
    /**
     * Any class that derives from AbstractLoss needs to define a `calculateLoss` method
     */
    template<class T>
    class AbstractLoss{
    public:
        AbstractLoss() = default;
        virtual ValuePtr operator()(const T&, const T&) = 0;
    };
}

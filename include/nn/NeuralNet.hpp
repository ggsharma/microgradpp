#pragma once

#include <memory>

#include "core/CoreReLU.hpp"
#include "core/CoreLinear.hpp"

namespace microgradpp::nn{

    template <class... T>
    std::unique_ptr<microgradpp::core::CoreLinear> Linear(T&&... args) {
        return std::make_unique<microgradpp::core::CoreLinear>(std::forward<T>(args)...);
    }
    std::unique_ptr<microgradpp::core::CoreReLU> ReLU(){
        return std::make_unique<microgradpp::core::CoreReLU>();
    }

}
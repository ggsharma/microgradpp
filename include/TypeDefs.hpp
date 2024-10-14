/**
 *  @file TypeDefs.h
 *  @brief Contains type definitions and preprocessor directives for the microgradpp project.
 *
 *  This file is part of the microgradpp project, a lightweight C++ library for neural
 *  network training and inference.
 *
 *  @section License
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 *  @section Author
 *  Gautam Sharma
 *  Email: gautamsharma2813@gmail.com
 *  Date: July 9, 2024
 *
 *  @details
 *  This header file contains strongly typed namespaces and preprocessor directives
 *  that are used throughout the microgradpp library. It serves as a central location
 *  for defining key macros and types that enhance code readability and maintainability.
 */

#ifndef MICROGRADPP_TYPEDEFS_H
#define MICROGRADPP_TYPEDEFS_H

// Strongly typed namespaces for the microgradpp project
namespace microgradpp::type {
    // Define types or enumerations related to microgradpp here
}

// Preprocessor directives
#define __MICROGRADPP_NO_DISCARD__ [[nodiscard]] /**< Attribute to indicate that the return value of a function should not be discarded. */
#define __MICROGRADPP_CLEAR__ microgradpp::Autograd::clear(); /**< Macro to invoke the clear function in the Autograd module. */

#endif //MICROGRADPP_TYPEDEFS_H

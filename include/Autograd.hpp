#pragma once
#include <vector>
#include <functional>


namespace microgradpp{
    class Value;
    using ValuePtr = std::shared_ptr<Value>;

    // TapeEntry
    struct TapeEntry {
        ValuePtr output;                          // The output of the operation
        std::function<void()> backward_fn;        // Function to compute gradients during backward pass
    };

    class Autograd {
    public:
        Autograd() = default;
        std::vector<TapeEntry> tape;              // Stores the sequence of operations

        // Add an operation to the tape
        void add_entry(const ValuePtr& output, std::function<void()> backward_fn) {
            tape.push_back({output, backward_fn});
        }

        // Backward pass: Traverse the tape in reverse order
        void backward() {
            for (auto it = tape.rbegin(); it != tape.rend(); ++it) {
                it->backward_fn();                 // Execute each backward function
            }
        }

        void clear() noexcept{
            global_tape.tape.clear();
        }
        static Autograd global_tape;
    };
}

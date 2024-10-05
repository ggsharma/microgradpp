#pragma once
#include <vector>
#include <functional>


namespace microgradpp{
    class Value;
    using ValuePtr = std::shared_ptr<Value>;

    // TapeEntry
    struct TapeEntry {
        ValuePtr output;                          // The output of the operation
        std::function<void()> backward_fn = nullptr;        // Function to compute gradients during backward pass
    };

    class Autograd {
    public:
        Autograd() = default;
        std::vector<TapeEntry> tape;              // Stores the sequence of operations
        std::unordered_map<size_t, std::function<void()>> TapeMap;

        // Add an operation to the tape
        void add_entry(ValuePtr& output, std::function<void()> backward_fn) {
            tape.push_back({output, std::move(backward_fn)});
        }

        // Backward pass: Traverse the tape in reverse order
        void backward() {
            for (auto it = global_tape.tape.rbegin(); it != global_tape.tape.rend(); ++it) {
                //if (it->output) {
                    if(it->backward_fn){
                        it->backward_fn();
                    }
//                } else {
//                    std::cerr << "Output is nullptr" << std::endl;
//                }

                // Execute each backward function
            }
        }

        static void clear() noexcept{
            global_tape.tape.clear();
        }

        static Autograd global_tape;
    };
}

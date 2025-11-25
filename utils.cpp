#include "utils.h"
#include <iostream>

namespace DaisoML {

void log(const std::string& message) {
    // In a real application, this could write to a file, etc.
    std::cout << "[LOG] " << message << std::endl;
}

} // namespace DaisoML

#ifndef DAISOML_UTILS_H
#define DAISOML_UTILS_H

#include <string>
#include <stdexcept>

namespace DaisoML {

// A simple logging utility
void log(const std::string& message);

// A custom exception class for our application
class DaisoException : public std::runtime_error {
public:
    explicit DaisoException(const std::string& message) : std::runtime_error(message) {}
};

// Add more utility function declarations here
// For example, file reading, timing, etc.

} // namespace DaisoML

#endif //DAISOML_UTILS_H

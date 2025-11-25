#ifndef DAISOML_SAMPLER_H
#define DAISOML_SAMPLER_H

#include "tensor.h"
#include <random>

namespace DaisoML {

// The Sampler is responsible for choosing the next token from the model's output logits.
class Sampler {
public:
    explicit Sampler(int vocab_size, float temperature = 0.8f, float top_p = 0.9f);

    // Sample a token from the logits tensor
    int sample(Tensor& logits);

private:
    int vocab_size;
    float temperature;
    float top_p;

    std::mt19937 rng; // Random number generator
};

} // namespace DaisoML

#endif //DAISOML_SAMPLER_H

#include "rmsnorm.h"
#include "../utils.h"
#include <cmath>


namespace DaisoML {

RMSNorm::RMSNorm(int dim) {
    weights = new Tensor({(size_t)dim});
    log("Initialized RMSNorm Layer.");
}

RMSNorm::~RMSNorm() {
    delete weights;
}

#include <cmath>

void RMSNorm::forward(Tensor& out, const Tensor& input) {
    if (input.shape() != out.shape() || input.size() != weights->size()) {
        throw DaisoException("RMSNorm shape mismatch.");
    }

    const float* x = input.data();
    float* y = out.data();
    const float* w = weights->data();
    size_t size = input.size();
    const float epsilon = 1e-5f;

    // 1. Calculate sum of squares
    float ss = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        ss += x[i] * x[i];
    }
    ss /= size;
    ss += epsilon;
    ss = 1.0f / std::sqrt(ss);

    // 2. Normalize and scale
    for (size_t i = 0; i < size; ++i) {
        y[i] = w[i] * (ss * x[i]);
    }
}


Tensor* RMSNorm::get_weights() {
    return weights;
}

} // namespace DaisoML

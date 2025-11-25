#include "feed_forward.h"
#include "../utils.h"
#include <fstream>
#include <vector>
#include <cmath>


namespace DaisoML {

// Helper to read a tensor from a file stream
static void read_tensor(std::ifstream& file, Tensor* tensor) {
    file.read(reinterpret_cast<char*>(tensor->data()), tensor->size() * sizeof(float));
}

FeedForward::FeedForward(int dim, int hidden_dim) {
    w1 = new Tensor({(size_t)hidden_dim, (size_t)dim});
    w2 = new Tensor({(size_t)dim, (size_t)hidden_dim});
    w3 = new Tensor({(size_t)hidden_dim, (size_t)dim});
    log("Initialized FeedForward (SwiGLU) Layer.");
}

FeedForward::~FeedForward() {
    delete w1;
    delete w2;
    delete w3;
}

void FeedForward::read_weights(std::ifstream& file) {
    read_tensor(file, w1);
    read_tensor(file, w2);
    read_tensor(file, w3);
}


#include <vector>
#include <cmath>

void FeedForward::forward(Tensor& out, const Tensor& input) {
    // This implements the SwiGLU logic: F(x) = (Swish(x @ w1) * (x @ w3)) @ w2
    // where Swish(x) = x * sigmoid(x)
    // The weights are transposed compared to some implementations.
    // Here: w1, w3 are (hidden_dim, dim), w2 is (dim, hidden_dim)
    // input is (dim), output is (dim)

    const auto& x = input.data();
    const auto& w1_data = w1->data();
    const auto& w2_data = w2->data();
    const auto& w3_data = w3->data();
    auto out_data = out.data();

    const int dim = input.shape()[0];
    const int hidden_dim = w1->shape()[0];

    // Temporary buffer for the hidden state
    std::vector<float> h(hidden_dim);
    std::vector<float> h_gate(hidden_dim);

    // 1. Calculate h = w1 @ x
    for (int i = 0; i < hidden_dim; ++i) {
        float val = 0.0f;
        for (int j = 0; j < dim; ++j) {
            val += w1_data[i * dim + j] * x[j];
        }
        h[i] = val;
    }

    // 2. Calculate h_gate = w3 @ x
    for (int i = 0; i < hidden_dim; ++i) {
        float val = 0.0f;
        for (int j = 0; j < dim; ++j) {
            val += w3_data[i * dim + j] * x[j];
        }
        h_gate[i] = val;
    }

    // 3. Apply SwiGLU activation
    for (int i = 0; i < hidden_dim; ++i) {
        float val = h[i];
        // Swish
        val *= (1.0f / (1.0f + std::exp(-val)));
        // Multiply by gate
        val *= h_gate[i];
        h[i] = val;
    }

    // 4. Project back down: out = w2 @ h
    for (int i = 0; i < dim; ++i) {
        float val = 0.0f;
        for (int j = 0; j < hidden_dim; ++j) {
            val += w2_data[i * hidden_dim + j] * h[j];
        }
        out_data[i] = val;
    }
}


} // namespace DaisoML

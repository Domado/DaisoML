#include "embedding.h"
#include "../utils.h"
#include <cstring>


namespace DaisoML {

Embedding::Embedding(int vocab_size, int dim) {
    weights = new Tensor({(size_t)vocab_size, (size_t)dim});
    log("Initialized Embedding Layer.");
}

Embedding::~Embedding() {
    delete weights;
}

#include <cstring> // For memcpy

void Embedding::forward(Tensor& out, const Tensor& tokens) {
    // Input `tokens` is a single integer ID.
    // `out` is the destination vector of size `dim`.
    // `weights` is the embedding table of size `vocab_size x dim`.
    if (tokens.size() != 1) {
        throw DaisoException("Embedding forward pass expects a single token ID.");
    }
    if (out.shape().size() != 1) {
        throw DaisoException("Embedding output must be a 1D vector.");
    }

    int token_id = static_cast<int>(tokens.data()[0]);
    size_t dim = out.shape()[0];
    size_t vocab_size = weights->shape()[0];

    if (token_id < 0 || (size_t)token_id >= vocab_size) {
        throw DaisoException("Token ID out of vocabulary bounds.");
    }
    if (dim != weights->shape()[1]) {
        throw DaisoException("Embedding output dimension mismatch.");
    }

    // Copy the embedding vector for the token ID into the output tensor
    const float* source = weights->data() + token_id * dim;
    float* dest = out.data();
    std::memcpy(dest, source, dim * sizeof(float));
}


Tensor* Embedding::get_weights() {
    return weights;
}

} // namespace DaisoML

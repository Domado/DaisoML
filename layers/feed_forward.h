#ifndef DAISOML_FEED_FORWARD_H
#define DAISOML_FEED_FORWARD_H

#include "../tensor.h"

namespace DaisoML {

// Also known as the SwiGLU layer in Llama models.
class FeedForward {
public:
    FeedForward(int dim, int hidden_dim);
    ~FeedForward();

    void forward(Tensor& out, const Tensor& input);
    void read_weights(std::ifstream& file);

private:
    Tensor* w1; // Corresponds to the gate projection
    Tensor* w2; // Corresponds to the down projection
    Tensor* w3; // Corresponds to the up projection
};

} // namespace DaisoML

#endif //DAISOML_FEED_FORWARD_H

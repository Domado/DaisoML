#ifndef DAISOML_EMBEDDING_H
#define DAISOML_EMBEDDING_H

#include "../tensor.h"

namespace DaisoML {

class Embedding {
public:
    Embedding(int vocab_size, int dim);
    ~Embedding();

    // Perform the embedding lookup
    void forward(Tensor& out, const Tensor& tokens);

    // Get a pointer to the weights tensor
    Tensor* get_weights();

private:
    Tensor* weights;

};

} // namespace DaisoML

#endif //DAISOML_EMBEDDING_H

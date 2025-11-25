#ifndef DAISOML_RMSNORM_H
#define DAISOML_RMSNORM_H

#include "../tensor.h"

namespace DaisoML {

class RMSNorm {
public:
    explicit RMSNorm(int dim);
    ~RMSNorm();

    // Perform the normalization
    void forward(Tensor& out, const Tensor& input);

    // Get a pointer to the weights tensor
    Tensor* get_weights();

private:
    Tensor* weights; // aka "gamma"

};

} // namespace DaisoML

#endif //DAISOML_RMSNORM_H

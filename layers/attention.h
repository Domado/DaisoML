#ifndef DAISOML_ATTENTION_H
#define DAISOML_ATTENTION_H

#include "../tensor.h"

namespace DaisoML {

class Attention {
public:
    Attention(int dim, int n_heads, int n_kv_heads, int seq_len);
    ~Attention();

    void forward(Tensor& out, const Tensor& input, int pos, int layer_idx, Tensor& full_k_cache, Tensor& full_v_cache);
    void read_weights(std::ifstream& file);

private:
    int dim;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int seq_len;

    // Weight matrices for Q, K, V and the output projection
    Tensor* wq;
    Tensor* wk;
    Tensor* wv;
    Tensor* wo;
};

} // namespace DaisoML

#endif //DAISOML_ATTENTION_H

#ifndef DAISOML_MODEL_H
#define DAISOML_MODEL_H

#include <string>
#include <vector>
#include "tensor.h"
#include "tokenizer.h"

#include "file_format.h"

namespace DaisoML {

// Forward declarations for layer types
class Embedding;
class RMSNorm;
class Attention;
class FeedForward;

struct TransformerBlock {
    RMSNorm* rms_att;
    Attention* attention;
    RMSNorm* rms_ffn;
    FeedForward* ffn;
};

class Model {
public:
    explicit Model(const std::string& path);
    ~Model();

    std::vector<int> generate(const std::vector<int>& tokens, int steps);
    Tokenizer& getTokenizer();

private:
    Tensor* forward(int token_id, int pos);

    void load_weights(const std::string& path);

    DaisoModelHeader config;
    Tokenizer tokenizer;

    // Model weights and layers
    Embedding* token_embedding_table;
    std::vector<TransformerBlock> layers;
    RMSNorm* rms_final;
    Tensor* final_weights; // (vocab_size, dim)

    // Key-value cache
    Tensor* k_cache;
    Tensor* v_cache;

    // Buffers for forward pass
    Tensor* x;
    Tensor* xb;
    Tensor* logits;
};




} // namespace DaisoML

#endif //DAISOML_MODEL_H

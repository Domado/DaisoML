#ifndef DAISOML_FILE_FORMAT_H
#define DAISOML_FILE_FORMAT_H

#include <cstdint>

namespace DaisoML {

// This header defines the binary format for DaisoML model files.

// The file starts with this magic number to identify it as a DaisoML file.
constexpr uint32_t DAISO_MAGIC = 0x64616973; // "dais" in ASCII

struct DaisoModelHeader {
    uint32_t magic;
    int32_t version;

    // Model configuration
    int32_t dim;        // transformer dimension
    int32_t hidden_dim; // for ffn layers
    int32_t n_layers;   // number of layers
    int32_t n_heads;    // number of query heads
    int32_t n_kv_heads; // number of key/value heads
    int32_t vocab_size; // vocabulary size
    int32_t seq_len;    // max sequence length

    // For simplicity, we can add other flags here if needed
    // For example, int32_t shared_weights;
};

// The layout of the file is:
// 1. DaisoModelHeader
// 2. Tokenizer vocabulary (if not part of the main model weights)
// 3. Model weights (Tensors) in a predefined order:
//    - token_embedding_table
//    - rms_att_weight for each layer
//    - wq, wk, wv, wo for each layer
//    - rms_ffn_weight for each layer
//    - w1, w2, w3 for each layer
//    - rms_final_weight
//    - final_weights (output projection)

} // namespace DaisoML

#endif // DAISOML_FILE_FORMAT_H

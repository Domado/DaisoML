#include "model.h"
#include "utils.h"
#include "file_format.h"
#include "sampler.h"
#include "layers/embedding.h"
#include "layers/rmsnorm.h"
#include "layers/attention.h"
#include "layers/feed_forward.h"

#include <fstream>
#include <memory>
#include <vector>

namespace DaisoML {

// Helper function to read a tensor's data from the file
void read_tensor(std::ifstream& file, Tensor* tensor) {
    file.read(reinterpret_cast<char*>(tensor->data()), tensor->size() * sizeof(float));
}

Model::Model(const std::string& path) {
    log("Initializing model from: " + path);
    load_weights(path);
    tokenizer = Tokenizer(config.vocab_size);
    log("Model initialization complete.");
}

Model::~Model() {
    log("Destroying model and freeing resources...");
    delete token_embedding_table;
    delete rms_final;
    delete final_weights;
    delete k_cache;
    delete v_cache;
    delete x;
    delete xb;
    delete logits;
    for (auto& block : layers) {
        delete block.rms_att;
        delete block.attention;
        delete block.rms_ffn;
        delete block.ffn;
    }
    log("Model destroyed.");
}

void Model::load_weights(const std::string& path) {
    log("Loading model weights from " + path);
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw DaisoException("Could not open model file: " + path);
    }

    // Read and validate header
    file.read(reinterpret_cast<char*>(&config), sizeof(DaisoModelHeader));
    if (config.magic != DAISO_MAGIC) throw DaisoException("Invalid model file: magic number mismatch.");
    if (config.version != 1) throw DaisoException("Unsupported model file version.");
    log("Model config loaded: dim=" + std::to_string(config.dim) + ", n_layers=" + std::to_string(config.n_layers));

    // Allocate layers and weights
    token_embedding_table = new Embedding(config.vocab_size, config.dim);
    layers.reserve(config.n_layers);
    for (int i = 0; i < config.n_layers; ++i) {
        layers.push_back({
            new RMSNorm(config.dim),
            new Attention(config.dim, config.n_heads, config.n_kv_heads, config.seq_len),
            new RMSNorm(config.dim),
            new FeedForward(config.dim, config.hidden_dim)
        });
    }
    rms_final = new RMSNorm(config.dim);
    final_weights = new Tensor({(size_t)config.vocab_size, (size_t)config.dim});
    
    // Allocate caches and buffers
    k_cache = new Tensor({(size_t)config.n_layers, (size_t)config.seq_len, (size_t)config.dim});
    v_cache = new Tensor({(size_t)config.n_layers, (size_t)config.seq_len, (size_t)config.dim});
    x = new Tensor({(size_t)config.dim});
    xb = new Tensor({(size_t)config.dim});
    logits = new Tensor({(size_t)config.vocab_size});

    // Read weights from the file
    log("Reading weights from file...");
    read_tensor(file, token_embedding_table->get_weights());
    for (int i = 0; i < config.n_layers; ++i) {
        read_tensor(file, layers[i].rms_att->get_weights());
        layers[i].attention->read_weights(file);
        read_tensor(file, layers[i].rms_ffn->get_weights());
        layers[i].ffn->read_weights(file);
    }
    read_tensor(file, rms_final->get_weights());
    read_tensor(file, final_weights);
    log("All weights loaded into memory.");
    file.close();
}

Tensor* Model::forward(int token_id, int pos) {
    // 1. Get token embedding
    Tensor token_tensor({1});
    token_tensor.data()[0] = (float)token_id;
    token_embedding_table->forward(*x, token_tensor);

    // 2. Forward through transformer blocks
    for (int i = 0; i < config.n_layers; ++i) {
        // RMSNorm before attention
        layers[i].rms_att->forward(*xb, *x);

        // Attention
        layers[i].attention->forward(*xb, *xb, pos, i, *k_cache, *v_cache);
        
        // Residual connection
        add(*x, *x, *xb);

        // RMSNorm before FFN
        layers[i].rms_ffn->forward(*xb, *x);

        // FFN
        layers[i].ffn->forward(*xb, *xb);

        // Residual connection
        add(*x, *x, *xb);
    }

    // 3. Final RMSNorm
    rms_final->forward(*x, *x);

    // 4. Classifier: calculate logits
    float* logits_data = logits->data();
    const float* x_data = x->data();
    const float* w_data = final_weights->data();
    for (int i = 0; i < config.vocab_size; ++i) {
        float val = 0.0f;
        for (int j = 0; j < config.dim; ++j) {
            val += w_data[i * config.dim + j] * x_data[j];
        }
        logits_data[i] = val;
    }

    return logits;
}


std::vector<int> Model::generate(const std::vector<int>& prompt_tokens, int steps) {
    log("Starting text generation...");
    std::vector<int> generated_tokens = prompt_tokens;
    
    Sampler sampler(config.vocab_size, 0.8f, 0.9f);

    int current_pos = 0;
    if (!prompt_tokens.empty()) {
        log("Processing prompt...");
        for (int token : prompt_tokens) {
            forward(token, current_pos);
            current_pos++;
        }
        log("Prompt processing finished.");
    }
    
    int next_token = prompt_tokens.empty() ? 0 : prompt_tokens.back(); // Start with last prompt token or 0

    log("Generating new tokens...");
    for (int i = 0; i < steps; ++i) {
        if (current_pos >= config.seq_len) {
            log("Reached max sequence length.");
            break;
        }

        forward(next_token, current_pos);
        Tensor* current_logits = logits;

        next_token = sampler.sample(*current_logits);
        generated_tokens.push_back(next_token);
        current_pos++;
    }
    
    log("Generation finished.");
    return generated_tokens;
}

Tokenizer& Model::getTokenizer() {
    return tokenizer;
}

} // namespace DaisoML

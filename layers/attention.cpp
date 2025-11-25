#include "attention.h"
#include "../utils.h"
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring> // For memcpy

namespace DaisoML {

// Helper to read a tensor from a file stream
static void read_tensor(std::ifstream& file, Tensor* tensor) {
    file.read(reinterpret_cast<char*>(tensor->data()), tensor->size() * sizeof(float));
}

// Helper to apply Rotary Position Embedding
void apply_rope(float* q, float* k, int pos, int head_dim) {
    for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / std::pow(10000.0f, (float)i / head_dim);
        float val = pos * freq;
        float fcr = std::cos(val);
        float fci = std::sin(val);

        float q0 = q[i];
        float q1 = q[i+1];
        q[i]   = q0 * fcr - q1 * fci;
        q[i+1] = q0 * fci + q1 * fcr;

        float k0 = k[i];
        float k1 = k[i+1];
        k[i]   = k0 * fcr - k1 * fci;
        k[i+1] = k0 * fci + k1 * fcr;
    }
}

Attention::Attention(int dim, int n_heads, int n_kv_heads, int seq_len)
    : dim(dim), n_heads(n_heads), n_kv_heads(n_kv_heads), seq_len(seq_len) {
    
    head_dim = dim / n_heads;

    wq = new Tensor({(size_t)dim, (size_t)dim});
    wk = new Tensor({(size_t)dim, (size_t)dim});
    wv = new Tensor({(size_t)dim, (size_t)dim});
    wo = new Tensor({(size_t)dim, (size_t)dim});

    log("Initialized Attention Layer.");
}

Attention::~Attention() {
    delete wq;
    delete wk;
    delete wv;
    delete wo;
}

void Attention::read_weights(std::ifstream& file) {
    read_tensor(file, wq);
    read_tensor(file, wk);
    read_tensor(file, wv);
    read_tensor(file, wo);
}

void Attention::forward(Tensor& out, const Tensor& input, int pos, int layer_idx, Tensor& full_k_cache, Tensor& full_v_cache) {
    const float* x = input.data();
    float* out_data = out.data();

    // Buffers for Q, K, V and the concatenated head outputs
    std::vector<float> q(dim);
    std::vector<float> k(dim);
    std::vector<float> v(dim);
    std::vector<float> y(dim); // Buffer for concatenated head outputs

    // 1. Calculate Q, K, V
    // q = wq @ x
    for (int i = 0; i < dim; ++i) {
        float val = 0.0f;
        for (int j = 0; j < dim; ++j) { val += wq->data()[i * dim + j] * x[j]; }
        q[i] = val;
    }
    // k = wk @ x
    for (int i = 0; i < dim; ++i) {
        float val = 0.0f;
        for (int j = 0; j < dim; ++j) { val += wk->data()[i * dim + j] * x[j]; }
        k[i] = val;
    }
    // v = wv @ x
    for (int i = 0; i < dim; ++i) {
        float val = 0.0f;
        for (int j = 0; j < dim; ++j) { val += wv->data()[i * dim + j] * x[j]; }
        v[i] = val;
    }

    // 2. Apply RoPE to Q and K heads
    for (int h = 0; h < n_heads; ++h) {
        float* q_head = &q[h * head_dim];
        float* k_head = &k[h * head_dim];
        apply_rope(q_head, k_head, pos, head_dim);
    }

    // 3. Save K and V to cache
    // Cache layout: [n_layers, seq_len, dim]
    size_t k_cache_offset = layer_idx * (seq_len * dim) + pos * dim;
    size_t v_cache_offset = layer_idx * (seq_len * dim) + pos * dim;

    std::memcpy(full_k_cache.data() + k_cache_offset, k.data(), dim * sizeof(float));
    std::memcpy(full_v_cache.data() + v_cache_offset, v.data(), dim * sizeof(float));

    // 4. Multi-head attention
    std::vector<float> scores(seq_len); // Max possible scores
    for (int h = 0; h < n_heads; ++h) {
        float* q_head = &q[h * head_dim];
        float* y_head = &y[h * head_dim];

        // Calculate attention scores
        for (int t = 0; t <= pos; ++t) {
            float* k_head_cached = (full_k_cache.data() + layer_idx * (seq_len * dim) + t * dim) + h * head_dim;
            float score = 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                score += q_head[i] * k_head_cached[i];
            }
            scores[t] = score / std::sqrt((float)head_dim);
        }

        // Softmax the scores
        float max_score = scores[0];
        for (int t = 1; t <= pos; ++t) { if (scores[t] > max_score) max_score = scores[t]; }
        float score_sum = 0.0f;
        for (int t = 0; t <= pos; ++t) {
            scores[t] = std::exp(scores[t] - max_score);
            score_sum += scores[t];
        }
        for (int t = 0; t <= pos; ++t) { scores[t] /= score_sum; }

        // Weighted sum of values
        std::fill(y_head, y_head + head_dim, 0.0f);
        for (int t = 0; t <= pos; ++t) {
            float* v_head_cached = (full_v_cache.data() + layer_idx * (seq_len * dim) + t * dim) + h * head_dim;
            for (int i = 0; i < head_dim; ++i) {
                y_head[i] += scores[t] * v_head_cached[i];
            }
        }
    }

    // 5. Final projection
    for (int i = 0; i < dim; ++i) {
        float val = 0.0f;
        for (int j = 0; j < dim; ++j) {
            val += wo->data()[i * dim + j] * y[j];
        }
        out_data[i] = val;
    }
}

} // namespace DaisoML

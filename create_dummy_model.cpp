#include "file_format.h"
#include "tensor.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <random>

// This utility creates a dummy model file with random weights.
// It's used for testing the model loading functionality of the main application.

void write_tensor(std::ofstream& file, const DaisoML::Tensor& tensor) {
    file.write(reinterpret_cast<const char*>(tensor.data()), tensor.size() * sizeof(float));
}

int main() {
    // 1. Define the model configuration
    DaisoML::DaisoModelHeader header = {
        .magic = DaisoML::DAISO_MAGIC,
        .version = 1,
        .dim = 288,
        .hidden_dim = 768,
        .n_layers = 6,
        .n_heads = 6,
        .n_kv_heads = 6,
        .vocab_size = 1024,
        .seq_len = 256
    };

    std::cout << "DaisoML Dummy Model Creator" << std::endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << "Dimensions:" << std::endl;
    std::cout << "  dim: " << header.dim << std::endl;
    std::cout << "  hidden_dim: " << header.hidden_dim << std::endl;
    std::cout << "  n_layers: " << header.n_layers << std::endl;
    std::cout << "  n_heads: " << header.n_heads << std::endl;
    std::cout << "  vocab_size: " << header.vocab_size << std::endl;
    std::cout << "  seq_len: " << header.seq_len << std::endl;
    std::cout << "---------------------------" << std::endl;

    // 2. Open the output file
    const char* filename = "dummy_model.bin";
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return 1;
    }

    // 3. Write the header
    file.write(reinterpret_cast<const char*>(&header), sizeof(DaisoML::DaisoModelHeader));
    std::cout << "Wrote header to " << filename << std::endl;

    // 4. Write random tensor data in the correct order
    std::cout << "Writing random tensor data..." << std::endl;

    // Use a random number generator to create somewhat realistic weights
    std::mt19937 rng(0); // Seed for reproducibility
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    auto create_random_tensor = [&](const std::vector<size_t>& shape) {
        DaisoML::Tensor t(shape);
        for (size_t i = 0; i < t.size(); ++i) {
            t.data()[i] = dist(rng);
        }
        return t;
    };

    // token_embedding_table
    auto token_embedding_table = create_random_tensor({(size_t)header.vocab_size, (size_t)header.dim});
    write_tensor(file, token_embedding_table);
    std::cout << "  - Wrote token_embedding_table" << std::endl;

    // Per-layer weights
    for (int i = 0; i < header.n_layers; ++i) {
        auto rms_att_weight = create_random_tensor({(size_t)header.dim});
        write_tensor(file, rms_att_weight);

        auto wq = create_random_tensor({(size_t)header.dim, (size_t)header.dim});
        write_tensor(file, wq);
        auto wk = create_random_tensor({(size_t)header.dim, (size_t)header.dim});
        write_tensor(file, wk);
        auto wv = create_random_tensor({(size_t)header.dim, (size_t)header.dim});
        write_tensor(file, wv);
        auto wo = create_random_tensor({(size_t)header.dim, (size_t)header.dim});
        write_tensor(file, wo);

        auto rms_ffn_weight = create_random_tensor({(size_t)header.dim});
        write_tensor(file, rms_ffn_weight);

        auto w1 = create_random_tensor({(size_t)header.hidden_dim, (size_t)header.dim});
        write_tensor(file, w1);
        auto w2 = create_random_tensor({(size_t)header.dim, (size_t)header.hidden_dim});
        write_tensor(file, w2);
        auto w3 = create_random_tensor({(size_t)header.hidden_dim, (size_t)header.dim});
        write_tensor(file, w3);
        std::cout << "  - Wrote weights for layer " << i << std::endl;
    }

    // Final weights
    auto rms_final_weight = create_random_tensor({(size_t)header.dim});
    write_tensor(file, rms_final_weight);
    std::cout << "  - Wrote rms_final_weight" << std::endl;

    // Output projection (optional, can be shared with embedding table)
    auto final_weights = create_random_tensor({(size_t)header.vocab_size, (size_t)header.dim});
    write_tensor(file, final_weights);
    std::cout << "  - Wrote final_weights" << std::endl;


    file.close();
    std::cout << "---------------------------" << std::endl;
    std::cout << "Successfully created dummy model file: " << filename << std::endl;

    return 0;
}

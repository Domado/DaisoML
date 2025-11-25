#include "tokenizer.h"
#include "utils.h"

namespace DaisoML {

Tokenizer::Tokenizer() : _vocab_size(0) {
    log("Empty tokenizer created.");
}

Tokenizer::Tokenizer(int vocab_size) : _vocab_size(vocab_size) {
    log("Tokenizer initialized with vocab size: " + std::to_string(vocab_size));
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    // This is a dummy implementation. It doesn't actually tokenize.
    log("Encoding text: '" + text + "' (dummy implementation).");
    std::vector<int> tokens;
    for (char c : text) {
        tokens.push_back(static_cast<int>(c)); // Simple char-to-int
    }
    return tokens;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    // This is a dummy implementation.
    log("Decoding tokens (dummy implementation).");
    std::string text;
    for (int token : tokens) {
        text += static_cast<char>(token);
    }
    return text;
}

int Tokenizer::vocab_size() const {
    return _vocab_size;
}

} // namespace DaisoML

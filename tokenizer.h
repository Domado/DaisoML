#ifndef DAISOML_TOKENIZER_H
#define DAISOML_TOKENIZER_H

#include <vector>
#include <string>

namespace DaisoML {

// A placeholder for a tokenizer. A real implementation would be much more complex,
// likely using something like Byte-Pair Encoding (BPE) or SentencePiece.
class Tokenizer {
public:
    Tokenizer();
    explicit Tokenizer(int vocab_size);


    // Convert text to a sequence of token IDs
    std::vector<int> encode(const std::string& text) const;

    // Convert a sequence of token IDs to text
    std::string decode(const std::vector<int>& tokens) const;

    int vocab_size() const;

private:
    // In a real scenario, this would be a map from string to int.
    // We'll use a simple placeholder.
    int _vocab_size;
};

} // namespace DaisoML

#endif //DAISOML_TOKENIZER_H

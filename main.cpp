#include <iostream>
#include <string>
#include <vector>
#include "model.h"
#include "tokenizer.h" // Include tokenizer for direct use if needed

int main(int argc, char **argv) {
    std::cout << "Welcome to DaisoML!" << std::endl;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];
    std::cout << "Loading model from: " << model_path << std::endl;

    try {
        DaisoML::Model model(model_path);
        std::cout << "Model loaded successfully." << std::endl;

        // Define a simple prompt
        std::string prompt_text = "Hello, my name is";
        std::cout << "Prompt: \"" << prompt_text << "\"" << std::endl;

        // Encode the prompt (dummy implementation for now)
        std::vector<int> prompt_tokens = model.getTokenizer().encode(prompt_text);
        std::cout << "Prompt tokens (dummy): ";
        for (int token : prompt_tokens) {
            std::cout << token << " ";
        }
        std::cout << std::endl;

        // Generate text
        int steps_to_generate = 50;
        std::vector<int> generated_tokens = model.generate(prompt_tokens, steps_to_generate);

        // Decode and print the generated text
        std::string generated_text = model.getTokenizer().decode(generated_tokens);
        std::cout << "Generated text: \"" << generated_text << "\"" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

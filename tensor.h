#ifndef DAISOML_TENSOR_H
#define DAISOML_TENSOR_H

#include <vector>
#include <cstddef>
#include <memory>

namespace DaisoML {

// A basic multi-dimensional Tensor class for floating point numbers.
class Tensor {
public:
    // Constructors
    Tensor();
    explicit Tensor(const std::vector<size_t>& shape);

    // Get the shape of the tensor
    const std::vector<size_t>& shape() const;

    // Get the total number of elements
    size_t size() const;

    // Get a pointer to the raw data
    float* data();
    const float* data() const;

    // Reshape the tensor (must have the same total size)
    void reshape(const std::vector<size_t>& new_shape);

    // Basic element access (for 1D, 2D, 3D for simplicity)
    float& at(size_t i);
    const float& at(size_t i) const;
    float& at(size_t i, size_t j);
    const float& at(size_t i, size_t j) const;
    float& at(size_t i, size_t j, size_t k);
    const float& at(size_t i, size_t j, size_t k) const;


private:
    std::vector<size_t> _shape;
    std::shared_ptr<std::vector<float>> _data;
    size_t _size;
};

// Tensor operations (will be implemented in tensor.cpp)
void matmul(Tensor& out, const Tensor& a, const Tensor& b);
void add(Tensor& out, const Tensor& a, const Tensor& b);
void softmax(Tensor& out, const Tensor& a);
void sigmoid(Tensor& out, const Tensor& a);
void element_wise_mul(Tensor& out, const Tensor& a, const Tensor& b);


} // namespace DaisoML

#endif //DAISOML_TENSOR_H

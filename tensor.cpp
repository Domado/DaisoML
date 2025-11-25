#include "tensor.h"
#include "utils.h"
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace DaisoML {

Tensor::Tensor() : _size(0) {}

Tensor::Tensor(const std::vector<size_t>& shape) : _shape(shape) {
    _size = 1;
    for (size_t dim : _shape) {
        if (dim == 0) { // Cannot have a dimension of size 0
             throw DaisoException("Tensor dimensions cannot be 0.");
        }
        _size *= dim;
    }
    _data = std::make_shared<std::vector<float>>(_size, 0.0f);
}

const std::vector<size_t>& Tensor::shape() const {
    return _shape;
}

size_t Tensor::size() const {
    return _size;
}

float* Tensor::data() {
    return _data->data();
}

const float* Tensor::data() const {
    return _data->data();
}

void Tensor::reshape(const std::vector<size_t>& new_shape) {
    size_t new_size = 1;
    for (size_t dim : new_shape) {
        new_size *= dim;
    }
    if (new_size != _size) {
        throw DaisoException("Cannot reshape: total size must remain the same.");
    }
    _shape = new_shape;
}

float& Tensor::at(size_t i) {
    if (_shape.size() != 1) throw DaisoException("at(i) requires a 1D tensor.");
    return (*_data)[i];
}
const float& Tensor::at(size_t i) const {
    if (_shape.size() != 1) throw DaisoException("at(i) requires a 1D tensor.");
    return (*_data)[i];
}

float& Tensor::at(size_t i, size_t j) {
    if (_shape.size() != 2) throw DaisoException("at(i, j) requires a 2D tensor.");
    return (*_data)[i * _shape[1] + j];
}
const float& Tensor::at(size_t i, size_t j) const {
    if (_shape.size() != 2) throw DaisoException("at(i, j) requires a 2D tensor.");
    return (*_data)[i * _shape[1] + j];
}

float& Tensor::at(size_t i, size_t j, size_t k) {
    if (_shape.size() != 3) throw DaisoException("at(i, j, k) requires a 3D tensor.");
    return (*_data)[i * _shape[1] * _shape[2] + j * _shape[2] + k];
}
const float& Tensor::at(size_t i, size_t j, size_t k) const {
    if (_shape.size() != 3) throw DaisoException("at(i, j, k) requires a 3D tensor.");
    return (*_data)[i * _shape[1] * _shape[2] + j * _shape[2] + k];
}


// Implementations for tensor operations

void matmul(Tensor& out, const Tensor& a, const Tensor& b) {
    // Naive matrix multiplication for 2D tensors
    if (a.shape().size() != 2 || b.shape().size() != 2 || out.shape().size() != 2) {
        throw DaisoException("Matmul currently only supports 2D tensors.");
    }
    const auto& a_shape = a.shape();
    const auto& b_shape = b.shape();
    const auto& out_shape = out.shape();

    if (a_shape[1] != b_shape[0] || out_shape[0] != a_shape[0] || out_shape[1] != b_shape[1]) {
        throw DaisoException("Matmul shape mismatch.");
    }

    for (size_t i = 0; i < a_shape[0]; ++i) {
        for (size_t j = 0; j < b_shape[1]; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < a_shape[1]; ++k) {
                sum += a.at(i, k) * b.at(k, j);
            }
            out.at(i, j) = sum;
        }
    }
}

void add(Tensor& out, const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape() || a.shape() != out.shape()) {
        throw DaisoException("Tensor addition shape mismatch.");
    }
    for (size_t i = 0; i < a.size(); ++i) {
        out.data()[i] = a.data()[i] + b.data()[i];
    }
}

void softmax(Tensor& out, const Tensor& a) {
    if (a.shape() != out.shape()) {
        throw DaisoException("Softmax shape mismatch.");
    }
    
    // Softmax is applied on the last dimension
    size_t last_dim = a.shape().back();
    size_t outer_size = a.size() / last_dim;

    for(size_t i = 0; i < outer_size; ++i) {
        const float* a_ptr = a.data() + i * last_dim;
        float* out_ptr = out.data() + i * last_dim;

        // Find max for numerical stability
        float max_val = *std::max_element(a_ptr, a_ptr + last_dim);

        // Exp and sum
        float sum = 0.0f;
        for(size_t j = 0; j < last_dim; ++j) {
            out_ptr[j] = std::exp(a_ptr[j] - max_val);
            sum += out_ptr[j];
        }

        // Normalize
        for(size_t j = 0; j < last_dim; ++j) {
            out_ptr[j] /= sum;
        }
    }
}

void sigmoid(Tensor& out, const Tensor& a) {
    if (a.shape() != out.shape()) {
        throw DaisoException("Sigmoid shape mismatch.");
    }
    for (size_t i = 0; i < a.size(); ++i) {
        out.data()[i] = 1.0f / (1.0f + std::exp(-a.data()[i]));
    }
}

void element_wise_mul(Tensor& out, const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape() || a.shape() != out.shape()) {
        throw DaisoException("Element-wise multiplication shape mismatch.");
    }
    for (size_t i = 0; i < a.size(); ++i) {
        out.data()[i] = a.data()[i] * b.data()[i];
    }
}

} // namespace DaisoML

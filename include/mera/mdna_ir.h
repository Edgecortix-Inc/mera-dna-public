/*
 * Copyright 2022 EdgeCortix Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MDNA_IR_H
#define MDNA_IR_H

#include <sstream>
#include <string>
#include <vector>

#include "nop/serializer.h"

namespace mera {
namespace ir {

enum class DataType { UInt8, Int8, Int32, Float32 };

static constexpr struct {
  DataType t;
  const char* name;
} type_map[] = {
    {DataType::UInt8, "UInt8"},
    {DataType::Int8, "Int8"},
    {DataType::Int32, "Int32"},
    {DataType::Float32, "Float32"},
};

struct Shape {
  std::vector<int> shape;
  int rank;
  int size;
  NOP_STRUCTURE(Shape, shape, rank, size);
};

// attributes
struct Dilations {
  int h;
  int w;
  NOP_STRUCTURE(Dilations, h, w);
};

struct Padding {
  int top;
  int bottom;
  int left;
  int right;
  NOP_STRUCTURE(Padding, top, bottom, left, right);
};

struct Strides {
  int h;
  int w;
  NOP_STRUCTURE(Strides, h, w);
};

struct Tensor {
  DataType type{};
  Shape shape{};
  std::string id;
  NOP_STRUCTURE(Tensor, type, shape, id);
};

struct Var {
  // outputs
  Tensor output;

  NOP_STRUCTURE(Var, output);
};

struct FloatVecConstant {
  // attributes
  std::vector<float> values;

  // outputs
  Tensor output;

  NOP_STRUCTURE(FloatVecConstant, values, output);
};

struct Int32VecConstant {
  // attributes
  std::vector<int32_t> values;

  // outputs
  Tensor output;

  NOP_STRUCTURE(Int32VecConstant, values, output);
};

struct Int8VecConstant {
  // attributes
  std::vector<int8_t> values;

  // outputs
  Tensor output;

  NOP_STRUCTURE(Int8VecConstant, values, output);
};

struct ReLU {
  // inputs
  Tensor input;

  // outputs
  Tensor output;

  NOP_STRUCTURE(ReLU, input, output);
};

struct AddOp {
  // inputs
  Tensor lhs;
  Tensor rhs;

  // outputs
  Tensor output;

  NOP_STRUCTURE(AddOp, lhs, rhs, output);
};

struct Quantize {
  // inputs
  Tensor input;

  Tensor output_scale;
  Tensor output_zero_point;

  int axis;

  // outputs
  Tensor output;

  NOP_STRUCTURE(Quantize, input, output_scale, output_zero_point, axis, output);
};

struct Dequantize {
  // inputs
  Tensor input;

  Tensor input_scale;
  Tensor input_zero_point;

  // outputs
  Tensor output;

  NOP_STRUCTURE(Dequantize, input, input_scale, input_zero_point, output);
};

struct Conv2d {
  Dilations dilations;
  Padding padding;
  Strides strides;

  int groups;
  int output_channels;

  // inputs
  Tensor input;
  Tensor weight;

  // outputs
  Tensor output;

  NOP_STRUCTURE(Conv2d, dilations, padding, strides, groups, output_channels,
                input, weight, output);
};

struct Clip {
  // attributes
  float min_value;
  float max_value;

  // inputs
  Tensor input;

  // outputs
  Tensor output;

  NOP_STRUCTURE(Clip, min_value, max_value, input, output);
};

struct QuantizedConv2d {
  Dilations dilations;
  Padding padding;
  Strides strides;

  int groups;
  int output_channels;

  // inputs
  Tensor input;
  Tensor weight;

  Tensor input_scale;
  Tensor input_zero_point;
  Tensor weight_scale;
  Tensor weight_zero_point;

  // outputs
  Tensor output;

  NOP_STRUCTURE(QuantizedConv2d, dilations, padding, strides, groups,
                output_channels, input, weight, input_scale, input_zero_point,
                weight_scale, weight_zero_point, output);

  // methods
  inline int GetInputChannels() const {
    return weight.shape.shape[1];
  }

  inline bool IsDepthwiseConv() const {
    return groups > 1 && groups == output_channels && GetInputChannels() == 1;
  }

  inline bool IsGroupConv() const {
    return groups > 1 && !IsDepthwiseConv();
  }

  inline bool IsPointwiseConv() const {
    return groups == 1;
  }
};

struct QuantizedAdd {
  // inputs
  Tensor lhs;
  Tensor rhs;

  Tensor lhs_scale;
  Tensor lhs_zero_point;
  Tensor rhs_scale;
  Tensor rhs_zero_point;
  Tensor output_scale;
  Tensor output_zero_point;

  // outputs
  Tensor output;

  NOP_STRUCTURE(QuantizedAdd, lhs, rhs, lhs_scale, lhs_zero_point, rhs_scale,
                rhs_zero_point, output_scale, output_zero_point, output);
};

struct QuantizedMul {
  // inputs
  Tensor lhs;
  Tensor rhs;

  Tensor lhs_scale;
  Tensor lhs_zero_point;
  Tensor rhs_scale;
  Tensor rhs_zero_point;
  Tensor output_scale;
  Tensor output_zero_point;

  // outputs
  Tensor output;

  NOP_STRUCTURE(QuantizedMul, lhs, rhs, lhs_scale, lhs_zero_point, rhs_scale,
                rhs_zero_point, output_scale, output_zero_point, output);
};

struct Requantize {
  Tensor input;
  Tensor input_scale;
  Tensor input_zero_point;
  Tensor output_scale;
  Tensor output_zero_point;
  Tensor output;
  NOP_STRUCTURE(Requantize, input, input_scale, input_zero_point, output_scale,
                output_zero_point, output);
};

struct BiasAdd {
  Tensor data;
  Tensor bias;
  Tensor output;
  NOP_STRUCTURE(BiasAdd, data, bias, output);
};

struct Cast {
  Tensor input;
  Tensor output;
  NOP_STRUCTURE(Cast, input, output);
};

struct Pad {
  Tensor input;
  Padding pad_width;
  double pad_value;
  Tensor output;
  NOP_STRUCTURE(Pad, input, pad_width, pad_value, output);
};

struct Upsampling {
  Tensor input;
  Tensor input_scale;
  Tensor input_zero_point;
  std::string method;
  std::string coordinate_transformation_mode;
  Tensor output;
  NOP_STRUCTURE(Upsampling, input, input_scale, input_zero_point, method,
                coordinate_transformation_mode, output);
};

struct UpsamplingFp {
  Tensor input;
  std::string method;
  std::string coordinate_transformation_mode;
  Tensor output;
  NOP_STRUCTURE(UpsamplingFp, input, method, coordinate_transformation_mode, output);
};

struct MaxPool2d {
  Tensor input;
  int pool_height;
  int pool_width;
  Strides strides;
  Padding padding;
  Tensor output;
  NOP_STRUCTURE(MaxPool2d, input, pool_height, pool_width, strides, padding,
                output);
};

struct LeakyReLU {
  Tensor input;
  Tensor input_scale;
  Tensor input_zero_point;
  Tensor output_scale;
  Tensor output_zero_point;
  double negative_slope;
  Tensor output;
  NOP_STRUCTURE(LeakyReLU, input, input_scale, input_zero_point, output_scale,
                output_zero_point, negative_slope, output);
};

struct SiLU {
  Tensor input;
  Tensor input_scale;
  Tensor input_zero_point;
  Tensor sigmoid_scale;
  Tensor sigmoid_zero_point;
  Tensor output_scale;
  Tensor output_zero_point;
  Tensor output;
  NOP_STRUCTURE(SiLU, input, input_scale, input_zero_point, sigmoid_scale,
                sigmoid_zero_point, output_scale, output_zero_point, output);
};

struct HSwish {
  Tensor input;
  Tensor input_scale;
  Tensor input_zero_point;
  Tensor output_scale;
  Tensor output_zero_point;
  Tensor output;
  NOP_STRUCTURE(HSwish, input, input_scale, input_zero_point, output_scale,
                output_zero_point, output);
};

struct Concatenate {
  std::vector<Tensor> inputs;
  int axis;
  Tensor output;
  NOP_STRUCTURE(Concatenate, inputs, axis, output);
};

struct Fc {
  Tensor input;
  Tensor weights;
  Tensor input_scale;
  Tensor input_zero_point;
  Tensor weight_scale;
  Tensor weight_zero_point;
  Tensor bias;
  Tensor output_scale;
  Tensor output_zero_point;
  Tensor output;
  NOP_STRUCTURE(Fc, input, weights,
                input_scale, input_zero_point, weight_scale,
                weight_zero_point, bias, output_scale, output_zero_point, output);
};

struct AvgPooling2d {
  Tensor input;
  Tensor output;
  NOP_STRUCTURE(AvgPooling2d, input, output);
};

struct Mean {
  Tensor input;
  Tensor input_scale;
  Tensor input_zero_point;
  Tensor output_scale;
  Tensor output_zero_point;
  Tensor output;
  NOP_STRUCTURE(Mean, input, input_scale, input_zero_point,
                output_scale, output_zero_point, output);
};

struct OutputNode {
  std::vector<Tensor> outputs;
  NOP_STRUCTURE(OutputNode, outputs);
};

struct Graph {
  typedef nop::Variant<Var, FloatVecConstant, Int32VecConstant, ReLU, AddOp,
                       Quantize, Dequantize, Conv2d, Clip, QuantizedConv2d,
                       QuantizedAdd, QuantizedMul, Requantize, BiasAdd, Cast,
                       Pad, Int8VecConstant, Upsampling, OutputNode, MaxPool2d,
                       LeakyReLU, SiLU, HSwish, Fc, AvgPooling2d, Mean, Concatenate,
                       UpsamplingFp>
      Operator;

  std::vector<Operator> operators;

  template <class Op, class... Args>
  Tensor Add(const std::string& name, DataType type, const Shape& shape,
             Args&&... args) {
    Tensor result{type, shape, name + std::to_string(next_id_++)};
    operators.emplace_back(Op{std::forward<Args>(args)..., result});
    return result;
  }

  void AddOutput(const std::vector<ir::Tensor>& output_tensors) {
    operators.emplace_back(OutputNode{output_tensors});
  }

  Tensor AddFloatVec(const std::vector<float>& values) {
    int size = int(values.size());
    return Add<FloatVecConstant>("FloatVecConstant", DataType::Float32,
                                 {{size}, 1, size}, values);
  }

  Tensor AddInt32Vec(const std::vector<int32_t>& values) {
    int size = int(values.size());
    return Add<Int32VecConstant>("Int32VecConstant", DataType::Int32,
                                 {{size}, 1, size}, values);
  }

  Tensor AddInt8Vec(const std::vector<int8_t>& values) {
    int size = int(values.size());
    return Add<Int8VecConstant>("Int8VecConstant", DataType::Int8,
                                {{size}, 1, size}, values);
  }

  NOP_STRUCTURE(Graph, operators);
  int next_id_{0};
};

struct Module {
  std::map<std::string, Graph> functions;

  Graph& AddFunction(const std::string& name) {
    if (functions.count(name)) {
      throw std::logic_error("Function already exists in this Module: " + name);
    }
    return functions[name];
  }

  Graph& GetFunction(const std::string& name) { return functions.at(name); }

  NOP_STRUCTURE(Module, functions);
};

}  // namespace ir
}  // namespace mera

#endif  // MDNA_IR_H

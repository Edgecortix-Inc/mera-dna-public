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
#ifndef MDNA_IR_IO_H
#define MDNA_IR_IO_H

#include "mdna_ir.h"

namespace mera {
namespace ir {

struct PrintVisitor {
  std::ostream& os;
  template <class T>
  std::ostream& operator()(const T& n) {
    os << n;
    return os;
  }

  std::ostream& operator()(const nop::EmptyVariant& n) {
    throw std::logic_error("Found an empty variant");
  }
};

inline std::ostream& operator<<(std::ostream& os, const Graph::Operator& n) {
  return n.Visit(PrintVisitor{os});
}

inline std::ostream& operator<<(std::ostream& os, const Graph& g) {
  for (const auto& node : g.operators) {
    os << node << std::endl;
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const DataType& n) {
  os << "dtype(" << type_map[int(n)].name << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Shape& n) {
  os << "shape(rank=" << n.rank << ", dimensions=[";
  for (auto it = n.shape.begin(); it != n.shape.end(); ++it) {
    os << *it << (std::next(it) != n.shape.end() ? "x" : "]");
  }
  os << ", size=" << n.size << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Tensor& n) {
  os << "Tensor(id=" << n.id << ", " << n.type << ", " << n.shape << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Var& n) {
  os << "Var(output=" << n.output << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const FloatVecConstant& n) {
  os << "FloatConstant(output=" << n.output << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Int32VecConstant& n) {
  os << "Int32Constant(output=" << n.output << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Int8VecConstant& n) {
  os << "Int8Constant(output=" << n.output << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const ReLU& n) {
  os << "ReLU(input=" << n.input.id << ", output=" << n.output.id << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Clip& n) {
  os << "Clip(input=" << n.input.id << ", output=" << n.output.id
     << ", min=" << n.min_value << ", max=" << n.max_value << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const AddOp& n) {
  os << "Add(input=" << n.lhs.id << ", input2=" << n.rhs.id
     << ", output=" << n.output.id << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Quantize& n) {
  os << "Quantize(input=" << n.input.id << ", output=" << n.output.id
     << ", scale=" << n.output_scale.id << ", zero=" << n.output_zero_point.id
     << ")"
     << ", axis=" << n.axis;
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Dequantize& n) {
  os << "Dequantize(input=" << n.input.id << ", output=" << n.output.id
     << ", scale=" << n.input_scale.id << ", zero=" << n.input_zero_point.id
     << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Conv2d& n) {
  os << "Conv2d(input=" << n.input.id << ", weights=" << n.weight.id
     << ", output=" << n.output.id;
  os << ", dilations=[h=" << n.dilations.h << ",w" << n.dilations.w << "]";
  os << ", pad=[t=" << n.padding.top << ",b=" << n.padding.bottom
     << ",l=" << n.padding.left << ",r=" << n.padding.right << "]";
  os << ", srides=[h=" << n.strides.h << ",w=" << n.strides.w << "]";
  os << ", groups=" << n.groups;
  os << ", outputChannels=" << n.output_channels;
  os << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const QuantizedConv2d& n) {
  os << "QuantizedConv2d(input=" << n.input.id << ", weights=" << n.weight.id
     << ", output=" << n.output.id;
  os << ", dilations=[h=" << n.dilations.h << ",w" << n.dilations.w << "]";
  os << ", pad=[t=" << n.padding.top << ",b=" << n.padding.bottom
     << ",l=" << n.padding.left << ",r=" << n.padding.right << "]";
  os << ", srides=[h=" << n.strides.h << ",w=" << n.strides.w << "]";
  os << ", groups=" << n.groups;
  os << ", outputChannels=" << n.output_channels;
  os << ", input_scale=" << n.input_scale.id;
  os << ", input_zero_point=" << n.input_zero_point.id;
  os << ", weight_scale=" << n.weight_scale.id;
  os << ", weight_zero_point=" << n.weight_zero_point.id;
  os << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Requantize& n) {
  os << "Requantize(input=" << n.input.id << ", output=" << n.output.id;
  os << ", input_scale=" << n.input_scale.id;
  os << ", input_zero_point=" << n.input_zero_point.id;
  os << ", output_scale=" << n.output_scale.id;
  os << ", output_zero_point=" << n.output_zero_point.id;
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const BiasAdd& n) {
  os << "BiasAdd(data=" << n.data.id << ", bias=" << n.bias.id;
  os << ", output=" << n.output.id << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Cast& n) {
  os << "Cast(input=" << n.input.id << ", output=" << n.output.id;
  os << ", output dtype=" << static_cast<int>(n.output.type);
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Pad& n) {
  os << "Pad(input=" << n.input.id << ", output=" << n.output.id;
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const QuantizedAdd& n) {
  os << "QuantizedAdd(input=" << n.lhs.id << ", output=" << n.rhs.id;
  os << ", lhs_scale=" << n.lhs_scale.id;
  os << ", lhs_zero_point=" << n.lhs_zero_point.id;
  os << ", rhs_scale=" << n.rhs_scale.id;
  os << ", rhs_zero_point=" << n.rhs_zero_point.id;
  os << ", output_scale=" << n.output_scale.id;
  os << ", output_zero_point=" << n.output_zero_point.id;
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const QuantizedMul& n) {
  os << "QuantizedMul(input=" << n.lhs.id << ", output=" << n.rhs.id;
  os << ", lhs_scale=" << n.lhs_scale.id;
  os << ", lhs_zero_point=" << n.lhs_zero_point.id;
  os << ", rhs_scale=" << n.rhs_scale.id;
  os << ", rhs_zero_point=" << n.rhs_zero_point.id;
  os << ", output_scale=" << n.output_scale.id;
  os << ", output_zero_point=" << n.output_zero_point.id;
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Upsampling& n) {
  os << "Upsampling(input=" << n.input.id << ", output=" << n.output.id;
  os << ", input_scale=" << n.input_scale;
  os << ", input_zero_point=" << n.input_zero_point;
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const MaxPool2d& n) {
  os << "MaxPool2d(input=" << n.input.id << ", output=" << n.output.id;
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const LeakyReLU& n) {
  os << "LeakyReLU(input=" << n.input.id << ", output=" << n.output.id;
  os << ", input_scale=" << n.input_scale;
  os << ", input_zero_point=" << n.input_zero_point;
  os << ", output_scale=" << n.output_scale;
  os << ", output_zero_point=" << n.output_zero_point;
  os << ", negative_slope=" << n.negative_slope;
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const SiLU& n) {
  os << "SiLU(input=" << n.input.id << ", output=" << n.output.id;
  os << ", input_scale=" << n.input_scale;
  os << ", input_zero_point=" << n.input_zero_point;
  os << ", sigmoid_scale=" << n.sigmoid_scale;
  os << ", sigmoid_zero_point=" << n.sigmoid_zero_point;
  os << ", output_scale=" << n.output_scale;
  os << ", output_zero_point=" << n.output_zero_point;
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const HSwish& n) {
  os << "HSwish(input=" << n.input.id << ", output=" << n.output.id;
  os << ", input_scale=" << n.input_scale;
  os << ", input_zero_point=" << n.input_zero_point;
  os << ", output_scale=" << n.output_scale;
  os << ", output_zero_point=" << n.output_zero_point;
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const OutputNode& n) {
  os << "OutputNode:output ids=";
  for (auto tensor : n.outputs) {
    os << tensor.id << ", ";
  }
  return os;
}

}  // namespace ir
}  // namespace mera

#endif  // MDNA_IR_IO_H

/*
 * Copyright 2023 EdgeCortix Inc.
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
#ifndef MDNA_IR_SHAPE_H
#define MDNA_IR_SHAPE_H

#include <array>
#include <vector>
#include <nop/serializer.h>

namespace mera {
namespace ir {

/**
 * @brief Class container describing the layout of a particular shape
 */
struct Layout {
  std::vector<char> layout_values;

  NOP_STRUCTURE(Layout, layout_values);

  std::string AsStr() const {
    std::stringstream ss;
    for (const char &l : layout_values) {
      ss << l;
    }
    return ss.str();
  }

  /**
   * @brief Returns whether this shape has the layout dimension 'lay_val'.
   */
  bool HasDim(char lay_val) const {
    return std::find(layout_values.begin(), layout_values.end(), lay_val) != layout_values.end();
  }
};

inline bool operator==(const Layout &lhs, const Layout &rhs) {
  if (lhs.layout_values.size() != rhs.layout_values.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.layout_values.size(); ++i) {
    if (lhs.layout_values[i] != rhs.layout_values[i]) {
      return false;
    }
  }
  return true;
}

inline bool operator!=(const Layout &lhs, const Layout &rhs) {
  return !(lhs == rhs);
}

namespace layout {
  // List of commonly used layouts

  const static Layout OIHW{{'O', 'I', 'H', 'W'}};
  const static Layout x{{'x'}};
  const static Layout C{{'C'}};
  const static Layout W{{'W'}};
  const static Layout HW{{'H', 'W'}};
  const static Layout NHWC{{'N', 'H', 'W', 'C'}};
  const static Layout NCHW{{'N', 'C', 'H', 'W'}};
  const static Layout NHW{{'N', 'H', 'W'}};
  const static Layout NCW{{'N', 'C', 'W'}};
  const static Layout OIW{{'O', 'I', 'W'}};
} // namespace layout


/**
 * @brief Class describing the shape and dimensions of a Tensor.
 */
struct Shape {
  std::vector<int> shape;
  int rank;
  int size;
  Layout layout;
  NOP_STRUCTURE(Shape, shape, rank, size, layout);

  Shape(const std::vector<int> &shape, const Layout &layout):
    shape(shape),
    rank(shape.size()),
    size([](const std::vector<int> &s) -> int { int r = 1; for (const auto & s_val : s) { r *= s_val; } return r; }(shape)),
    layout(layout) {
    if ((size_t)rank != layout.layout_values.size()) {
      throw std::runtime_error("Incorrect rank (" + std::to_string(rank) + ") for layout " + layout.AsStr());
    }
  }

  Shape(std::initializer_list<int> shape, const Layout &layout): Shape(std::vector<int>(shape), layout) {}
  Shape() : Shape({1}, layout::x) {} // Default is single value

  int& at(std::vector<int>::size_type pos)  { return shape.at(pos); }
  const int& at(std::vector<int>::size_type pos) const { return shape.at(pos); }

  /**
   * @brief Returns the current dimension of the layout parameter 'lay_val'. Error if layout does not exist
   */
  int DimOf(char lay_val) const {
    if (shape.size() != layout.layout_values.size()) {
      throw std::runtime_error("Shape size (" + std::to_string(shape.size()) + ") does not match with layout "
        + layout.AsStr());
    }
    for (size_t i = 0; i < layout.layout_values.size(); ++i) {
      if (layout.layout_values[i] == lay_val) {
        return shape[i];
      }
    }
    throw std::runtime_error("Could not find layout value " + std::string(1, lay_val) + " in layout " + layout.AsStr());
  }

  /**
   * @brief Returns the current dimension of the layout parameter 'lay_val'. Otherwise returns the provided default_value.
   */
  int DimOf(char lay_val, int default_val) const {
    return HasDim(lay_val) ? DimOf(lay_val) : default_val;
  }

  /**
   * @brief Returns the axis position of the layout parameter 'lay_val'. Error if layout does not exist
   */
  int AxisOf(char lay_val) const {
    for (size_t i = 0; i < layout.layout_values.size(); ++i) {
      if (layout.layout_values[i] == lay_val) {
        return i;
      }
    }
    throw std::runtime_error("Could not find layout value " + std::string(1, lay_val) + " in layout " + layout.AsStr());
  }

  /**
   * @brief Pads the shape at the axis of layout parameter 'lay_val' to the next multiple of 'value'.
   */
  void PadDimTo(char lay_val, size_t value) {
    const int axis = AxisOf(lay_val);
    shape.at(axis) = ((shape.at(axis) + value - 1) / value) * value;
    int r = 1.0;
    for (const auto &s : shape) { r *= s; }
    size = r;
  }

  /**
   * @brief Return whether this shape contains the layout dimension 'lay_val'.
   */
  bool HasDim(char lay_val) const { return layout.HasDim(lay_val); }

  /**
   * @brief Unpacks the shape's dimensions into variables.
   */
  template<size_t N>
  std::array<int, N> Unpack() const {
    std::array<int, N> ret;
    if (rank != N) {
      throw std::runtime_error("Unpack size incorrect for rank " + std::to_string(rank) + ". Provided " + std::to_string(N));
    }
    for (int i = 0; i < N; ++i) {
      ret[i] = shape[i];
    }
    return ret;
  }

  /**
   * @brief Unpacks the shape's dimensions into variables with a certain layout, expanding with 1s for extra dimensions.
   * Provided layout has to encpasulate or expand the current layout.
   */
  template<size_t N>
  std::array<int, N> UnpackAs(const Layout &lay_ext) const {
    std::array<int, N> ret;
    if (lay_ext.layout_values.size() != N) {
      throw std::runtime_error("Unpack size incorrect for layout " + lay_ext.AsStr() + ". Expected " + std::to_string(N));
    }
    // Check all values of our layout exist in the provided layout
    for (const auto &l : layout.layout_values) {
      if (!lay_ext.HasDim(l)) {
        throw std::runtime_error("Unpack error: Layout " + layout.AsStr() + " cannot be unpacked into " + lay_ext.AsStr());
      }
    }
    for (int i = 0; i < N; ++i) {
      ret[i] = layout.HasDim(lay_ext.layout_values[i]) ? DimOf(lay_ext.layout_values[i]) : 1;
    }
    return ret;
  }

  /**
   * @brief Converts shape into equivalent shape with another layout
   */
  template<size_t N>
  Shape ReshapeAs(const Layout &lay_ext) const {
    auto d = UnpackAs<N>(lay_ext);
    return Shape{std::vector<int>(d.begin(), d.end()), lay_ext};
  }
};

inline bool operator==(const Shape& lhs, const Shape& rhs) {
  return lhs.shape == rhs.shape && lhs.rank == rhs.rank && lhs.size == rhs.size && lhs.layout == rhs.layout;
}

inline bool operator!=(const Shape& lhs, const Shape& rhs) { return !(lhs == rhs); }

const static Shape s_one = Shape({1}, layout::x);

} // namespace ir
} // namespace mera

#endif // MDNA_IR_SHAPE_H

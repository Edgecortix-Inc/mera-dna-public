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
#ifndef MDNA_IR_TYPE_H
#define MDNA_IR_TYPE_H

namespace mera {
namespace ir {

enum class DataType { UInt8, Int8, Int32, Float32, BrainFloat16 };

static constexpr struct {
  DataType t;
  const char* name;
  size_t size;
} type_map[] = {
    {DataType::UInt8, "UInt8", 1},
    {DataType::Int8, "Int8", 1},
    {DataType::Int32, "Int32", 4},
    {DataType::Float32, "Float32", 4},
    {DataType::BrainFloat16, "BrainFloat16", 2}
};

/**
 * @brief Returns the string representation of this type.
 */
inline std::string ToString(DataType t) { return std::string(type_map[(int)t].name); }

/**
 * @brief Returns the size, in bytes, of this type.
 */
inline size_t SizeOf(DataType t) { return type_map[(int)t].size; }

}  // namespace ir
}  // namespace mera

#endif // MDNA_IR_TYPE_H

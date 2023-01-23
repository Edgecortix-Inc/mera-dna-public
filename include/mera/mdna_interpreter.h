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
#ifndef MDNA_INTERPRETER_H
#define MDNA_INTERPRETER_H

#include <vector>
#include <string>
#include <optional>

#include "mdna_ir.h"

namespace mera {
namespace interpreter {

struct InterpreterBufInfo {
  const std::vector<int> shape;
  const int64_t size;
  const void *data;
  const ir::DataType type;
};

struct InterpreterNodeInfo {
  const std::string id;
  const std::string op_type;
};

struct Interpreter_ {
  virtual ~Interpreter_() {}

  virtual std::optional<InterpreterBufInfo> GetInterpreterBuffer(const std::string &id) const = 0;

  virtual std::vector<InterpreterNodeInfo> GetInterpreterNodeList() const = 0;
};

}
}

#endif // MDNA_INTERPRETER_H

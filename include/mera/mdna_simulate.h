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
#ifndef MDNA_SIMULATE_H
#define MDNA_SIMULATE_H

#include "nop/serializer.h"

namespace mera {
namespace simulate {

struct SimulationPack {
  std::vector<std::string> code;
  std::vector<uint8_t> parameters;
  NOP_STRUCTURE(SimulationPack, code, parameters);
};
using SimPackModule = std::map<std::string, SimulationPack>;

}  // namespace simulate
}  // namespace mera

#endif  // MDNA_SIMULATE_H

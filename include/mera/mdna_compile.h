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
#ifndef MDNA_COMPILE_H
#define MDNA_COMPILE_H

#include "mdna_ir.h"

namespace mera {
namespace compile {

std::vector<uint8_t> Compile(const mera::ir::Module& mod, std::string arch,
                             std::string ccfg);

}  // namespace compile
}  // namespace mera

#endif  // MDNA_COMPILE_H

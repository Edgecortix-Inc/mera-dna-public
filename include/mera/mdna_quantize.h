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
#ifndef MDNA_QUANTIZE_H
#define MDNA_QUANTIZE_H

#include <vector>
#include <string>
#include <memory>

#include "mdna_ir.h"
#include "mdna_interpreter.h"

namespace mera {
namespace quantizer {

struct Quantizer : public interpreter::Interpreter_ {
  virtual ~Quantizer() {}

  /**
   * @brief Resets the internal state of the observers.
   */
  virtual void Reset() = 0;

  /**
   * @brief Runs an individual calibration image with the data provided by args.
   */
  virtual void RunCalibrationImage(const std::vector<void*> &args) = 0;

  /**
   * @brief Get a JSON serialization list of all calculated quantization
   * parameters from this calibrated model.
   */
  virtual std::string CalculateQParams() = 0;

  /**
   * @brief Takes all the quantization parameters gathered during calibration and transforms
   * the model into a quantized one. Returns a serialized data representation of it.
   */
  virtual std::vector<uint8_t> QuantizeTransform() = 0;
};

std::unique_ptr<Quantizer> CreateQuantizer(const std::vector<uint8_t> &serialized_module);

ir::Module LoadMeraQuantizedModule(const std::vector<uint8_t> &transformed_module, const std::string &func_name); 

} // namespace quantizer
} // namespace mera

#endif // MDNA_QUANTIZE_H

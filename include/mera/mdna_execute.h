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
#ifndef MDNA_EXECUTE_H
#define MDNA_EXECUTE_H

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mera {
namespace execute {

struct ExecutorMetrics {
  enum class MetricsType {
    RUNTIME, POWER
  };

  ExecutorMetrics() {}
  virtual ~ExecutorMetrics() {}

  // Method for serializing up to TVM
  const std::string AsString(MetricsType type) const;
 protected:
  std::map<const std::string, std::string> metrics_;
  std::map<std::string, MetricsType> metric_types_;
};

class Executor {
 public:
  virtual ~Executor() {}
  virtual ExecutorMetrics Run(const std::string& function,
                              std::vector<void*>& args) const = 0;
};

enum class DeviceRunTarget {
  NONE = 0, /* Running on host */
  SAKURA_1 = 1,
  XILINX_U50 = 2,
  INTEL_IA420 = 3,
  ACHRONIX = 4
};

std::ostream &operator<<(std::ostream &os, const DeviceRunTarget &t);

std::unique_ptr<Executor> CreateExecutor(
    const std::vector<uint8_t>& serialized_module, DeviceRunTarget device_run_target);

ExecutorMetrics Execute(const Executor* executor, const std::string& function,
                        std::vector<void*>& args);

ExecutorMetrics Execute(const std::vector<uint8_t>& serialized_module,
                        const std::string& function, std::vector<void*>& args);

}  // namespace execute
}  // namespace mera

#endif  // MDNA_EXECUTE_H

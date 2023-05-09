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
#ifndef MDNA_BLOCKS_H
#define MDNA_BLOCKS_H

#include "mdna_ir.h"
#include <vector>
#include <memory>

/**
 * @file mdna_blocks.h
 * @brief Definition of the different MERA block implementations provided by mera-dna.
 */
namespace mera {
namespace blocks {

/**
 * @brief Base class for a MERA block.
 */
struct MeraBlock {
  virtual ~MeraBlock() {}

  /**
   * @brief Serialize this block parameters into a byte buffer
   */
  virtual std::vector<uint8_t> SaveParams() const = 0;

  /**
   * @brief Deserializes the byte buffer containing the block parameters and
   * assign their values to this object.
   */
  virtual void LoadParams(const std::vector<uint8_t> &params) = 0;

  /**
   * @brief run the MERA block with the provided IO data buffers.
   */
  virtual void Evaluate(const std::vector<void*> &buffers) const = 0;
};


struct Yolov5Post : public MeraBlock {
  Yolov5Post() = default;
  Yolov5Post(int batch, int num_classes, int img_h, int img_w);

  virtual std::vector<uint8_t> SaveParams() const override;

  virtual void LoadParams(const std::vector<uint8_t> &params) override;

  static std::string GetBlockId() { return "YOLOv5Post"; }

  virtual void Evaluate(const std::vector<void*> &buffers) const override;

  friend std::ostream &operator<<(std::ostream &os, const Yolov5Post &yolo_post);

  int batch;
  int num_classes;
  int img_h;
  int img_w;
};

struct Yolov5i8Post : public Yolov5Post {
  Yolov5i8Post() = default;
  Yolov5i8Post(int batch, int num_classes, int img_h, int img_w,
    const std::vector<float> &feat_scales, const std::vector<int32_t> &feat_zps);

  virtual std::vector<uint8_t> SaveParams() const override;

  virtual void LoadParams(const std::vector<uint8_t> &params) override;

  static std::string GetBlockId() { return "YOLOv5i8Post"; }

  virtual void Evaluate(const std::vector<void*> &buffers) const override;

  friend std::ostream &operator<<(std::ostream &os, const Yolov5i8Post &yolo_post);

  std::vector<float> feat_scales;
  std::vector<int32_t> feat_zps;
};

}  // namespace blocks
}  // namespace mera

#endif // MDNA_BLOCKS_H

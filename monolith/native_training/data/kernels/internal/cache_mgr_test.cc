// Copyright 2022 ByteDance and/or its affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "monolith/native_training/data/kernels/internal/cache_mgr.h"
#include <cstdlib>
#include "gtest/gtest.h"

#include "absl/strings/str_cat.h"

using NamedFeature = ::monolith::io::proto::NamedFeature;
using ChannelCache = ::monolith::io::proto::ChannelCache;

namespace tensorflow {
namespace monolith_tf {
namespace internal {
namespace {

static constexpr uint64_t MASK = (1L << 48) - 1;

void gen_named_feature(NamedFeature *nf) {
  int slot = std::rand() % 1024;
  nf->set_name(absl::StrCat("fc_", slot));
  auto *fid_v2_list = nf->mutable_feature()->mutable_fid_v2_list();
  int num_fids = std::abs(std::rand() % 20) + 1;
  for (int i = 0; i < num_fids; ++i) {
    uint64_t fid = ((uint64_t)slot << 48) | ((std::rand() % 100000) & MASK);
    fid_v2_list->add_value(fid);
  }
}

void gen_item_features(ItemFeatures *item) {
  int num_feats = std::abs(std::rand() % 20) + 1;
  for (int i = 0; i < num_feats; ++i) {
    NamedFeature nf;
    gen_named_feature(&nf);
    if (!item->example_features.contains(nf.name())) {
      item->example_features.insert({nf.name(), nf});
    }
  }
}

void fill_cache_with_gid(CacheWithGid *cwg) {
  for (int i = 0; i < 80; ++i) {
    std::shared_ptr<ItemFeatures> item = std::make_shared<ItemFeatures>();
    gen_item_features(item.get());

    int gid = std::abs(std::rand() % 1024) + 1;
    cwg->Push(gid, item);
  }
}

TEST(CACHE_MGR, CacheWithGid) {
  CacheWithGid cwg(100, 20);
  fill_cache_with_gid(&cwg);

  ChannelCache cache;
  cwg.ToProto(&cache);

  CacheWithGid cwg2(100, 20);
  cwg2.FromProto(cache);
}

TEST(CACHE_MGR, CacheManager) {
  CacheManager cm(1000, 20);

  for (int i = 0; i < 50; ++i) {
    const std::shared_ptr<ItemFeatures> item = std::make_shared<ItemFeatures>();
    gen_item_features(item.get());
    int gid = std::abs(std::rand() % 1024) + 1;
    cm.Push(1, gid, item);
  }
  EXPECT_EQ(cm.GetCache().size(), 1);
}

}  // namespace
}  // namespace internal
}  // namespace monolith_tf
}  // namespace tensorflow

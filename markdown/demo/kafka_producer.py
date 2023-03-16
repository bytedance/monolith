# Copyright 2022 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ml_dataset import get_preprocessed_dataset, serialize_one
from tqdm import tqdm
from kafka import KafkaProducer

if __name__ == "__main__":
  ds = get_preprocessed_dataset()
  producer = KafkaProducer(bootstrap_servers=['127.0.0.1:9092'])
  for count, val in tqdm(enumerate(ds), total=len(ds)):
    # note: we omit error callback here for performance
    producer.send(
      "movie-train", key=str(count).encode('utf-8'), value=serialize_one(val), headers=[])
  producer.flush()

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

from absl import app, flags, logging

from monolith.agent_service.agent import main as agent_main
from monolith.agent_service.agent_client import main as agent_client_main
from monolith.agent_service.tfs_client import main as tfs_client_main


FLAGS = flags.FLAGS
flags.DEFINE_enum("bin_name", "agent",
                  ["agent", "agent_client"],
                  "bin_name: agent, agent_client")

def main(_):
  if FLAGS.bin_name == 'agent':
    agent_main(_)
  elif FLAGS.bin_name == 'agent_client':
    agent_client_main(_)
  elif FLAGS.bin_name == 'tfs_client':
    tfs_client_main(_)
  else:
    raise ValueError("Unknown bin: {}".format(FLAGS.bin_name))


if __name__ == '__main__':
  app.run(main)

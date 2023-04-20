#!/bin/bash
set -eux

script_dir=`dirname $0`
abs_script_dir=`realpath $script_dir`

use_gpu="${1:-false}"

rm -rf output
mkdir -p output

bazel --version

if [ "$use_gpu" = "true" ]; then
  bazel build \
    --output_filter=DONT_MATCH_ANYTHING \
    --define=framework_shared_object=false \
    --config=cuda \
    @org_tensorflow_serving//tensorflow_serving/model_servers:tensorflow_model_server
else
  bazel build \
    --output_filter=DONT_MATCH_ANYTHING \
    --define=framework_shared_object=false \
    @org_tensorflow_serving//tensorflow_serving/model_servers:tensorflow_model_server
fi

# We can't compile archon in TensorFlow Py

bazel build \
  --output_filter=DONT_MATCH_ANYTHING \
  --define=framework_shared_object=false \
  //monolith/agent_service:agent

bazel build \
  --output_filter=DONT_MATCH_ANYTHING \
  --define=framework_shared_object=false \
  //monolith/agent_service:tfs_client

# 1) prepare output
mkdir -p output/bin/
mkdir -p output/lib/

function clear_external() {
  runfiles_dir="$1"
  echo "runfiles_dir: $runfiles_dir"
  pushd $runfiles_dir/__main__/external
  for external_name in `ls .`;
  do
    if [ -d "../../$external_name" ]; then
      rm -rf $external_name && ln -s ../../$external_name .
    fi
  done
  popd
}

cp -frL bazel-bin/monolith/agent_service/agent.runfiles/ output/
clear_external output/agent.runfiles

cp -frL bazel-bin/monolith/agent_service/tfs_client.runfiles/ output/
clear_external output/tfs_client.runfiles

cp -frL bazel-bin/external/org_tensorflow_serving/tensorflow_serving/model_servers/tensorflow_model_server.runfiles/ output/
rm output/tensorflow_model_server.runfiles/org_tensorflow_serving -rf

cd output/bin
ln -s ../agent.runfiles/__main__/monolith/agent_service/agent .
ln -s ../tfs_client.runfiles/__main__/monolith/agent_service/tfs_client .
ln -s ../tensorflow_model_server.runfiles/__main__/external/org_tensorflow_serving/tensorflow_serving/model_servers/tensorflow_model_server .
cd -

cp -rL $abs_script_dir/run_server output/bin/
cp -rL $abs_script_dir/conf output

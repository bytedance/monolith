#! /bin/bash

# set -x

# grpc port: PORT3
gpu_server_target="10.209.87.151:9469"  # multi "10.210.92.156:9361,10.198.98.198:9433"
cpu_server_target="10.211.69.228:9388"

tool_dir=`dirname $0`
abs_tool_dir=`realpath $tool_dir`
entry_agent_path="$abs_tool_dir/agent.conf"
profile_data_dir="$abs_tool_dir/profile_data"

bin_name="tfs_client"
bin_path="/home/lilintong.22222/.cache/bazel/_bazel_lilintong.22222/5282ccf6d1eb9e524c65d4bb4a5b4207/execroot/__main__/bazel-out/k8-opt/bin/monolith/agent_service/${bin_name}"


function run_pro() {
  target="$1"
  conf_path="$2"
  batch_size="$3"
  parallel_num="$4"
  profile_duration="$5"
  profile_data_dir="$6"

  $bin_path \
    --target=$target \
    --conf=$conf_path \
    --cmd_type="profile" \
    --input_type="example_batch" \
    --batch_size=$batch_size \
    --parallel_num=$parallel_num \
    --profile_duration=$profile_duration \
    --profile_data_dir=$profile_data_dir \
    --has_sort_id
}

function run_pro_async() {
  target="$1"
  conf_path="$2"
  batch_size="$3"
  parallel_num="$4"
  profile_duration="$5"
  profile_data_dir="$6"

  $bin_path \
    --target=$target \
    --conf=$conf_path \
    --cmd_type="profile" \
    --input_type="example_batch" \
    --batch_size=$batch_size \
    --parallel_num=$parallel_num \
    --profile_duration=$profile_duration \
    --profile_data_dir=$profile_data_dir \
    --has_sort_id &
}

function run_alg() {
  target="$1"
  conf_path="$2"
  batch_size="$3"
  input_path="$4"
  output_path="$5"

  $bin_path \
    --target=$target \
    --conf=$conf_path \
    --cmd_type="get" \
    --input_type="example_batch" \
    --batch_size=$batch_size \
    --input_file=$input_path \
    --has_sort_id > $output_path
}

function compare_alg() {
  a_server_target="$1"
  b_server_target="$1"

  rm -f input_alg.pb
  run_alg $a_server_target $entry_agent_path 1 input_alg.pb output_alg_gpu.txt
  run_alg $b_server_target $entry_agent_path 1 input_alg.pb output_alg_cpu.txt

  diff -urN output_alg_gpu.txt output_alg_cpu.txt > compare_alg.diff
  cat compare_alg.diff
}

function warmup() {
  server_target="$1"

  for ((i=0; i<3; i++)); do
    run_alg $server_target $entry_agent_path 1 input_alg.pb output_alg_gpu.txt
    cat output_alg_gpu.txt
  done
}

bazel build :${bin_name}

warmup $gpu_server_target

# compare_alg $gpu_server_target $cpu_server_target

# sync profile
run_pro $gpu_server_target $entry_agent_path 128 1 300 $profile_data_dir
# run_pro $cpu_server_target $entry_agent_path 128 1 600 $profile_data_dir

# async profile
for ((i=1; i<=6; i++)); do
  run_pro_async $gpu_server_target $entry_agent_path 128 11 300 $profile_data_dir
done

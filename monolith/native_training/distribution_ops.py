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

import logging
import os
from typing import List, Tuple, Optional, NamedTuple

import tensorflow as tf

from idl.matrix.proto.example_pb2 import FeatureConfigs
from monolith.native_training.runtime.ops import gen_monolith_ops

gen_distribution_ops = gen_monolith_ops


def split_by_indices(indices: tf.Tensor, tensor: tf.Tensor,
                     num_splits: int) -> tf.Tensor:
  """
  Split |input| elements in into |num_splits| tensors based on indices.
  |input| is treated as a list of tensors.
  """
  return gen_distribution_ops.monolith_split_by_indices(indices, tensor,
                                                        num_splits)


@tf.RegisterGradient("MonolithSplitByIndices")
def _split_by_indices_gradient(op: tf.Operation, *grads):
  indices = op.inputs[0]
  tensor = op.inputs[1]
  tensor_grad = gen_distribution_ops.monolith_split_by_indices_gradient(
      indices, tensor, grads)
  return None, tensor_grad


def ragged_split_by_indices(
    indices: tf.Tensor, num: tf.RaggedTensor,
    num_splits: int) -> Tuple[List[tf.RaggedTensor], List[tf.RaggedTensor]]:
  """Split a int64 ragged tensor into |num_splits| ragged tensor based on indices.
  Returns splitted ragged tensor and splitted original position of each number in ragged tensor.
  For example,
  indices = [0, 1, 0, 1]
  num = [[4, 3, 2], [1]]
  num_splits = 2
  ===>
  [
    [[4, 2], []],
    [[3], [1]],
  ],
  [
    [[0, 2], []],
    [[1], [3]],
  ]
  """
  splitted_num, splitted_num_splits, splitted_pos = gen_distribution_ops.monolith_ragged_split_by_indices(
      indices, num.values, num.row_splits, num_splits=num_splits)
  results = []
  pos = []
  for i in range(num_splits):
    results.append(
        tf.RaggedTensor.from_row_splits(splitted_num[i],
                                        splitted_num_splits[i],
                                        validate=False))
    pos.append(
        tf.RaggedTensor.from_row_splits(splitted_pos[i],
                                        splitted_num_splits[i],
                                        validate=False))
  return results, pos


class _UniqueKeyWithValueAndOffsetResult(NamedTuple):
  unique_key: tf.RaggedTensor
  value_offset: tf.RaggedTensor
  value_buffer: tf.Tensor


def unique_key_with_value_and_offset(key: tf.RaggedTensor,
                                     dims: List[int],
                                     generate_buffer=True):
  """Uniques the keys within each key[i], and generates the corresponding value offset map.
  For key[i][j], the coresponding value's length is dims[i].

  unique_key - the unique result of each key[i]
  value_buffer - a SharedTensor represents all values concated.
  value_offset - a ragged tensor with ragged_rank=2. value_offset[i][j] repensents the all offsets
  in value_buffer for unique_key[i][j]. So if we know the value of key[i][j] is v. We need to fill
  value_buffer[value_offset[i][j][0]:value_offset[i][j]+dims[i]] = v
  value_buffer[value_offset[i][j][1]:value_offset[i][j]+dims[i]] = v
  ...

  For example,
  key = [[0, 1, 0], [0]]
  dims = [2, 3]
  =>
  unique_key = [[0, 1], [0]]
  value_offset = [[[0, 4], [2]], [[6]]]
  value_buffer = float buffer with length 2*3 + 3*1 = 9
  """
  results = gen_distribution_ops.monolith_unique_key_with_value_and_offset(
      key.values, key.row_splits, dims=dims, generate_buffer=generate_buffer)
  return _UniqueKeyWithValueAndOffsetResult(
      unique_key=tf.RaggedTensor.from_row_splits(results[0],
                                                 results[1],
                                                 validate=False),
      value_offset=tf.RaggedTensor.from_nested_row_splits(
          results[2], [results[1], results[3]], validate=False),
      value_buffer=results[4])


def fill_with_offset_map(pos: tf.RaggedTensor, value: tf.Tensor,
                         value_offset_map: tf.RaggedTensor,
                         value_buffer: tf.Tensor, dims: List[int]) -> tf.Tensor:
  """Fill the |value| to |value_buffer| for each |pos| in |value_offset_map|.
  Specifically, for each pos[i][j], we extrac value slice from value (v),
  we got all positions for pos[i][j], which are value_offset_map.values[pos[i][j]][0],
  value_offset_map.values[pos[i][j]][1] ...
  And fill the value_buffer.

  For example,
  pos = [[0, 1], [2]]
  value = [0, 1, 2, 3, 4, 5, 6]
  value_offset_map = [[[0, 4], [2]], [[6]]]
  dims = [2, 3]
  =>
  value_buffer = [0, 1, 2, 3, 0, 1, 4, 5, 6]
  """
  value_offset_map_1d = value_offset_map.values
  return gen_distribution_ops.monolith_fill_with_offset_map(
      pos.values,
      pos.row_splits,
      value,
      value_offset_map_1d.values,
      value_offset_map_1d.row_splits,
      value_buffer,
      dims=dims,
  )


def fill_with_offset_map_gradient(pos: tf.RaggedTensor, grad: tf.Tensor,
                                  value_offset_map: tf.RaggedTensor,
                                  dims: List[int]) -> tf.Tensor:
  value_offset_map_1d = value_offset_map.values
  return gen_distribution_ops.monolith_fill_with_offset_map_gradient(
      pos.values,
      pos.row_splits,
      grad,
      value_offset_map_1d.values,
      value_offset_map_1d.row_splits,
      dims=dims,
  )


@tf.RegisterGradient("MonolithFillWithOffsetMap")
def _fill_with_offset_map_gradient(op: tf.Operation, grad):
  value_offset_map = tf.RaggedTensor.from_nested_row_splits(
      op.inputs[3], [op.inputs[1], op.inputs[4]], validate=False)
  pos = tf.RaggedTensor.from_row_splits(op.inputs[0], op.inputs[1])
  backprop_grad = fill_with_offset_map_gradient(pos,
                                                grad,
                                                value_offset_map,
                                                dims=op.get_attr("dims"))
  return None, None, backprop_grad, None, None, None


def finalize_shared_tensor(shared_tensor_handles: List[tf.Tensor], dtype,
                           shape):
  """Finalize a shared tensor and it won't be accessible in the future.
  shared_tensor_handles - the *same handle* which repeats several times.
  The reason why it is a list is to build a meaningful dependencies for output tensor,
  which is useful for gradient calculation.
  For example,
  t = SharedTensor() 
  t1 = FillPart(t, data0)
  t2 = FillPart(t, data1)
  t = finalize_shared_tensor([t1, t2])
  In this case, we want the grad on t can be propagated back to data0, data1.
  """
  return gen_distribution_ops.monolith_finalize_shared_tensor(
      shared_tensor_handles, dtype=dtype, shape=shape)


@tf.RegisterGradient("MonolithFinalizeSharedTensor")
def _finalize_shared_tensor_gradient(op: tf.Operation, grad):
  return grad


def reorder_by_indices(input: tf.Tensor, shard_ids: tf.Tensor,
                       num_of_shards: int) -> List[tf.Tensor]:
  """
  Reorder the input based on precomputed shard_ids from the caller.
  Example 1:
    input: [1, 2, 3, 2]
    shard_ids: [1, 0, 1, 0]
    num_of_shards: 2

    output => [2, 3, 1]
    shard_sizes => [1, 2]

  Args:
    input: 1-D int64/2-D float tensor with shape [N,]
    shard_ids: 1-D int32 tensor with shape [N], shard_ids[i] represents the shard is for input[i, ...]
    num_of_shards: a int32 scalar, representing the number of shards.
  Returns:
    Output: reordered 1-D int64/2-D float tensor with shape [M,], M<=N.
    Shard_sizes: 1-D int32 tensor with shape [num_of_shards].

  """
  return gen_distribution_ops.monolith_reorder_by_indices(
      input, shard_ids, num_of_shards)


def fused_reorder_by_indices(
    inputs: List[tf.Tensor], num_of_shards: int,
    dim_sizes: List[int]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """
  Reorder and dedup int64 values in a list of tensors according to the sharding.
  
  Note that the deduplication is applied per tensor of inputs. In other words, 
    the dedup process does not check duplicated ints across tensors of inputs.

  With dim_sizes of the embedding merged slot, it maps each fid to the offset 
    in the expected fused embedding to be generated based on the output fids.
    For more intuitive cases and explanations, check out the unit test cases.

  Examples:
      inputs: [[0, 1, 0], [3, 2, 3], [5, 6, 7]]
      num_of_shards: 2
      dim_sizes: [1, 2, 3]

      output => [0,2,6,1,3,5,7]
      shard_sizes => [3,4]
      sharded_slot_sizes => [1,1,1,1,1,2]
      fused_embedding_offsets => [[0,6,0],[7,1,7],[9,3,12]]
  
  Args:
    inputs: List of 1-D int64 tensors.
    num_of_shards: a int32 scalar, representing the number of shards.
  Returns:
    output: reordered 1-D int64 tensor.
    shard_sizes: 1-D int32 tensor with shape (num_of_shards).
    sharded_slot_sizes: 1-D int32 tensor with shape (num_of_shards * len(inputs)).
    fused_embedding_offsets: List of 1-d int32 tensor with shape as corresponding inputs.
  """
  # shard_indicies: List of 1-D int32 tensors, shard_indicies[i] represents for inputs[i]
  # We only trigger this N-1 sharding scheme when it is beyond single host mode.
  rank0_empty_shard = os.environ.get('MONOLITH_SYNC_EMPTY_RANK0_PS_SHARD',
                                     '1') == '1'
  if num_of_shards > 4 and rank0_empty_shard:
    # We add 1 as an offset for the index numbers, so shard 0 is free from this.
    shard_indicies = [
        tf.cast(tf.math.add(tf.ones(tf.shape(ids), dtype=tf.int64),
                            tf.math.floormod(ids, num_of_shards - 1)),
                dtype=tf.int32) for ids in inputs
    ]
  else:
    shard_indicies = [
        tf.cast(tf.math.floormod(ids, num_of_shards), dtype=tf.int32)
        for ids in inputs
    ]
  return gen_distribution_ops.fused_reorder_by_indices(inputs, shard_indicies,
                                                       num_of_shards, dim_sizes)
  # # An Alternative Implementation based on TensorFlow Builtin Ops:
  # with tf.name_scope('fused_reorder_by_indicies'):
  #   inputs = [tf.unique(ids)[0] for ids in inputs]
  #   shard_indicies = [tf.cast(tf.math.floormod(ids, num_of_shards), dtype=tf.int32) for ids in inputs]
  #   sharded_slot_lists = [tf.dynamic_partition(ids, indicies, num_of_shards) for ids, indicies in zip(inputs, shard_indicies)]
  #   outputs = []
  #   shard_sizes = []
  #   sharded_slot_sizes = []
  #   for i in range(num_of_shards):
  #     sizes_per_shard = []
  #     for m in range(len(inputs)):
  #       outputs.append(sharded_slot_lists[m][i])
  #       sizes_per_shard.append(tf.size(sharded_slot_lists[m][i]))
  #     shard_sizes.append(tf.reduce_sum(sizes_per_shard))
  #     sharded_slot_sizes.extend(sizes_per_shard)
  #   output = tf.concat(outputs, axis=0)
  # return output, tf.convert_to_tensor(shard_sizes), tf.convert_to_tensor(sharded_slot_sizes)


def map_id_to_embedding(ids: List[tf.Tensor],
                        embeddings: List[tf.Tensor],
                        input: tf.Tensor,
                        use_multi_threads: bool = True) -> tf.Tensor:
  """
  Map int64 in input to embedding. Output will have an extra dim at last
  which equals to embedding dim. The length of ids and embeddings must match.
  Args:
    ids: a list of 1-D int64 tensor.
    embeddings: a list of 2-D float32 tensor. Represents mapping.
    use_multi_threads: True if the caller wants to use multi-threads.
  """
  if len(ids) != len(embeddings):
    raise ValueError(
        "ids length and embeddings lenght must match. {} vs {}".format(
            len(ids), len(embeddings)))

  return gen_distribution_ops.monolith_map_id_to_embedding(
      ids, embeddings, input, use_multi_threads=use_multi_threads)


def fused_embedding_to_layout(
    embeddings_list: List[tf.Tensor],
    fid_list_row_split: List[tf.Tensor],
    fid_offset: tf.Tensor,
    feature_offset: tf.Tensor,
    nfl_offset: tf.Tensor,
    batch_size: tf.Tensor,
    variant_type: str,
    feature_cfgs: FeatureConfigs,
    ps_num: int,
    parallel_flag: int = 0,
    version: int = 2,
):
  assert variant_type in {
      'example', 'example_batch', 'examplebatch', 'instance'
  }
  variant_type = 'example_batch' if variant_type == 'examplebatch' else variant_type
  feature_cfgs_str = feature_cfgs.SerializeToString()
  N = 0
  for layout, conf in feature_cfgs.out_configs.items():
    N += len(conf.shape)
  if version == 2:
    layout_tensors = gen_distribution_ops.monolith_embedding_to_layout_v2(
        embeddings_list=embeddings_list,
        fid_list_row_split=fid_list_row_split,
        fid_offset=fid_offset,
        feature_offset=feature_offset,
        nfl_offset=nfl_offset,
        batch_size=batch_size,
        num_out=N,
        variant_type=variant_type,
        feature_cfgs=feature_cfgs_str,
        ps_num=ps_num,
        parallel_flag=parallel_flag)
  else:
    layout_tensors = gen_distribution_ops.monolith_embedding_to_layout(
        embeddings_list=embeddings_list,
        fid_offset=fid_offset,
        feature_offset=feature_offset,
        nfl_offset=nfl_offset,
        batch_size=batch_size,
        num_out=N,
        variant_type=variant_type,
        feature_cfgs=feature_cfgs_str)
  return layout_tensors


@tf.RegisterGradient("MonolithEmbeddingToLayout")
def _fused_embedding_to_layout_grad(op: tf.Operation, *grads):
  M = op.get_attr("M")  # fid_num
  embeddings_list = op.inputs[0:M]
  fid_offset, feature_offset, nfl_offset, batch_size = op.inputs[M], op.inputs[
      M + 1], op.inputs[M + 2], op.inputs[M + 3]
  variant_type = op.get_attr("variant_type")
  feature_cfgs_str = op.get_attr("feature_cfgs")
  embeddings_grad_list = gen_distribution_ops.monolith_embedding_to_layout_grad(
      embeddings_list=embeddings_list,
      fid_offset=fid_offset,
      feature_offset=feature_offset,
      nfl_offset=nfl_offset,
      batch_size=batch_size,
      tensors_grad=grads,
      variant_type=variant_type,
      feature_cfgs=feature_cfgs_str)
  return embeddings_grad_list + [None] * 4


@tf.RegisterGradient("MonolithEmbeddingToLayoutV2")
def _fused_embedding_to_layout_grad_v2(op: tf.Operation, *grads):
  M = op.get_attr("M")  # fid_num
  pre = 0
  embeddings_list = op.inputs[0:M]
  pre += M
  fid_list_row_split = op.inputs[pre:pre + M]
  pre += M
  fid_offset, feature_offset, nfl_offset, batch_size = op.inputs[
      pre], op.inputs[pre + 1], op.inputs[pre + 2], op.inputs[pre + 3]
  variant_type = op.get_attr("variant_type")
  feature_cfgs_str = op.get_attr("feature_cfgs")
  ps_num = op.get_attr("ps_num")
  embeddings_grad_list = gen_distribution_ops.monolith_embedding_to_layout_grad_v2(
      embeddings_list=embeddings_list,
      fid_list_row_split=fid_list_row_split,
      fid_offset=fid_offset,
      feature_offset=feature_offset,
      nfl_offset=nfl_offset,
      batch_size=batch_size,
      tensors_grad=grads,
      variant_type=variant_type,
      feature_cfgs=feature_cfgs_str,
      ps_num=ps_num,
      parallel_flag=0)
  return embeddings_grad_list + [None] * (M + 4)


def fused_embedding_to_layout_grad(nfl_offset: tf.Tensor,
                                   feature_offset: tf.Tensor,
                                   fid_offset: tf.Tensor,
                                   batch_size: tf.Tensor,
                                   embeddings_list: List[tf.Tensor],
                                   fid_list_row_split: List[tf.Tensor],
                                   layout_tensors_grad: List[tf.Tensor],
                                   variant_type: str,
                                   feature_cfgs: FeatureConfigs,
                                   ps_num: int,
                                   parallel_flag=0) -> List[tf.Tensor]:
  feature_cfgs_str = feature_cfgs.SerializeToString()
  assert variant_type in {
      'example', 'example_batch', 'examplebatch', 'instance'
  }
  variant_type = 'example_batch' if variant_type == 'examplebatch' else variant_type
  embeddings_grad_list = gen_distribution_ops.monolith_embedding_to_layout_grad_v2(
      embeddings_list=embeddings_list,
      fid_list_row_split=fid_list_row_split,
      fid_offset=fid_offset,
      feature_offset=feature_offset,
      nfl_offset=nfl_offset,
      batch_size=batch_size,
      tensors_grad=layout_tensors_grad,
      variant_type=variant_type,
      feature_cfgs=feature_cfgs_str,
      ps_num=ps_num,
      parallel_flag=parallel_flag)
  return embeddings_grad_list


@tf.RegisterGradient("MonolithMapIdToEmbedding")
def _map_id_to_embedding_gradient(op: tf.Operation, grads: tf.Tensor):
  num_splits = op.get_attr("num_splits")
  ids = [op.inputs[i] for i in range(num_splits)]
  input = op.inputs[2 * num_splits]
  embedding_grads = gen_distribution_ops.monolith_map_id_to_embedding_gradient(
      ids, input, grads)
  return [None] * num_splits + embedding_grads + [None]


def map_id_to_embedding_gradient_back_prop(ids: tf.Tensor, input: tf.Tensor,
                                           grads: tf.Tensor):
  """
  The manual back prop for MonolithMapIdToEmbedding.

  Returns:
    output: A list of 2-D tensors [K, dim], sum(K)=N

  """
  embedding_grads = gen_distribution_ops.monolith_map_id_to_embedding_gradient(
      ids, input, grads)
  return embedding_grads


def gather_embeddings_by_input(ids: tf.Tensor,
                               embeddings: tf.Tensor,
                               input: tf.Tensor,
                               use_multi_threads: bool = False) -> tf.Tensor:
  """
  Gather embeddings based on input with a shape [N] and an ids:embeddings map. 
  The ids with a shape [M] is mapped element-wise to embeddings with a shape [M, dim],
  e.g., for any index i, ids(i)'s embedding is embeddings(i).

  Example:
    ids: [1, 2, 3]
    embeddings: [[1., 1.], [2., 2.], [3., 3.]]
    input: [1, 3, 2, 3]

    output=>[[1., 1.], [3., 3.], [2., 2.], [3., 3.]]
    index_mapping=>[0, 2, 1, 2]
  
  Args:
    ids: a 1-D int64 tensor [M].
    embeddings: a 2-D float32 tensor [M, dim]. Mapped in order with ids.
    input: a int32 tensor with shape [N], N >= M. Input value is range from 0 to M-1.

  Returns:
    output: a 2-D tensor [N, dim].
    index_mapping: a 1-D tensor [N].

  """
  return gen_distribution_ops.monolith_gather_embeddings_by_input(
      ids, embeddings, input, use_multi_threads=use_multi_threads)


@tf.RegisterGradient("MonolithGatherEmbeddingsByInput")
def _gather_embeddings_by_ids_gradient(
    op: tf.Operation, grads: tf.Tensor,
    index_mapping_grads: Optional[tf.Tensor]):
  ids = op.inputs[0]
  index_mapping = op.outputs[1]
  embedding_grads = gen_distribution_ops.monolith_gather_embeddings_by_input_gradient(
      ids, grads, index_mapping)
  return [None, embedding_grads, None]


def fused_gather_embeddings_by_input(
    fused_embeddings: tf.Tensor, fused_embedding_offsets: List[tf.Tensor],
    embedding_dims: List[int]) -> List[tf.Tensor]:
  return gen_distribution_ops.monolith_fused_gather_embeddings_by_input(
      fused_embeddings, fused_embedding_offsets, embedding_dims=embedding_dims)


def fused_gather_embeddings_by_input_gradient(
    fused_embeddings: tf.Tensor,
    grads: List[tf.Tensor],
    embedding_offsets: List[tf.Tensor],
    embedding_dims: List[int],
) -> tf.Tensor:
  return gen_distribution_ops.monolith_fused_gather_embeddings_by_input_gradient(
      fused_embeddings, grads, embedding_offsets, embedding_dims=embedding_dims)


def reduce_mean(id_indices: tf.Tensor, id_values: tf.Tensor,
                id_length: tf.Tensor):
  """
  Very similar to tf.sparse.reduce_mean. The difference is now id_values is a 2-D
  tensors instead of 1-D tensor.
  Args:
    id_indices: 2-D tensor represents a list of positions of id_values.
    id_values: 2-D tensor which represents a list of actual values. (Value is 1-D tensor)
    id_length: should be a shape which equals to [batch_size] 
  """
  return gen_distribution_ops.monolith_reduce_mean(id_indices, id_values,
                                                   id_length)


def gather_embeddings_by_ids_gradient_back_prop(ids: tf.Tensor,
                                                grads: tf.Tensor,
                                                index_mapping: tf.Tensor):
  """
  The manual back prop for MonolithGatherEmbeddingsByInput.

  Returns:
    output: a 2-D tensor [N, dim].

  """
  embedding_grads = gen_distribution_ops.monolith_gather_embeddings_by_input_gradient(
      ids, grads, index_mapping)
  return embedding_grads


@tf.RegisterGradient("MonolithReduceMean")
def _reduce_mean_gradient(op: tf.Operation, grads: tf.Tensor):
  id_indices = op.inputs[0]
  id_value_grads = gen_distribution_ops.monolith_reduce_mean_gradient(
      id_indices, grads)
  return None, id_value_grads, None


def reduce_sum(id_indices: tf.Tensor,
               id_values: tf.Tensor,
               id_length: tf.Tensor,
               name=None):
  """
  Very similar to tf.sparse.reduce_sum. The difference is now id_values is a 2-D
  tensors instead of 1-D tensor.
  Args:
    id_indices: 2-D tensor represents a list of positions of id_values.
    id_values: 2-D tensor which represents a list of actual values. (Value is 1-D tensor)
    id_length: should be a shape which equals to [batch_size] 
  """
  return gen_distribution_ops.monolith_reduce_sum(id_indices,
                                                  id_values,
                                                  id_length,
                                                  name=name)


@tf.RegisterGradient("MonolithReduceSum")
def _reduce_sum_gradient(op: tf.Operation, grads: tf.Tensor):
  id_indices = op.inputs[0]
  id_value_grads = gen_distribution_ops.monolith_reduce_sum_gradient(
      id_indices, grads)
  return None, id_value_grads, None


def reduce_sqrtn(id_indices: tf.Tensor, id_values: tf.Tensor,
                 id_length: tf.Tensor):
  """
  Very similar to the combiner method in tf.tpu.experimental.embedding.TPUEmbedding
  The input is a 
  Args:
    id_indices: 2-D tensor represents a list of positions of id_values.
    id_values: 2-D tensor which represents a list of actual values. (Value is 1-D tensor)
    id_length: should be a shape which equals to [batch_size] 
  """
  return gen_distribution_ops.monolith_reduce_square_norm(
      id_indices, id_values, id_length)


@tf.RegisterGradient("MonolithReduceSquareNorm")
def _reduce_sum_gradient(op: tf.Operation, grads: tf.Tensor):
  id_indices = op.inputs[0]
  id_values = op.inputs[1]
  id_value_grads = gen_distribution_ops.monolith_reduce_square_norm_gradient(
      id_indices, id_values, grads)
  return None, id_value_grads, None


def fused_sorted_segment_sum(indices: List[tf.Tensor], values: List[tf.Tensor],
                             shapes: List[tf.Tensor]):
  """
  It combines multiple segment_sum into one GPU kernel.
  
  Args:
    indicies: List of Indices a.k.s 1-D SORTED segment ids
    values: List of Values to scatter into the output tensor.
    shapes: List of Shapes that Must have the same type as indices.
  Output:
    reduced: output tensors, i-the tensor has a shape `shapes[i]`.
  """
  return gen_distribution_ops.monolith_fused_segment_sum(
      indices, values, shapes)


@tf.RegisterGradient("MonolithFusedSegmentSum")
def _FusedSegmentSumGrad(op, *grads):
  n = len(grads)
  updates_grads = [
      tf.gather_nd(grad, tf.expand_dims(indices,
                                        -1))  # Similar to fused ScatterNd
      for grad, indices in zip(grads, op.inputs[:n])
  ]
  return [None] * n + updates_grads + [None] * n


def fused_reduce_sum_and_split(id_indices: tf.Tensor,
                               id_values: tf.Tensor,
                               id_length: tf.Tensor,
                               split_dims: List[int],
                               name: str = None):
  """
  Very similar to tf.sparse.reduce_sum. It combines with a fused split op.
  Args:
    id_indices: 1-D tensor represents a list of positions of id_values.
    id_values: 2-D tensor which represents a list of actual values. (Value is 1-D tensor)
    id_length: should be a shape which equals to "batch_size"
    split_dims: dimensions for the split vectors. Sum(split_dims)=id_values.dim(1)
  Output:
    reduced: M output tensors, and i-th tensor has a shape [bs, split_dims[i]].
  """
  id_indices = tf.expand_dims(id_indices, -1)
  id_length = tf.cast(tf.expand_dims(id_length, 0),
                      dtype=tf.int64)  # To remove cast support int32 for cpu
  num_of_splits = len(split_dims)
  return gen_distribution_ops.monolith_fused_reduce_sum_and_split(id_indices,
                                                                  id_values,
                                                                  id_length,
                                                                  num_of_splits,
                                                                  split_dims,
                                                                  name=name)


@tf.RegisterGradient("MonolithFusedReduceSumAndSplit")
def _fused_reduce_sum_and_split_gradient(op: tf.Operation, *grads):
  id_indices = op.inputs[0]
  split_dims = op.get_attr("split_dims")
  id_value_grads = gen_distribution_ops.monolith_fused_reduce_sum_and_split_gradient(
      id_indices, grads, split_dims=split_dims)
  return None, id_value_grads, None


def fused_reduce_and_split_gpu(splits: List[tf.Tensor],
                               embeddings: List[tf.Tensor],
                               slice_dims: List[List[int]],
                               name: str = None) -> List[tf.Tensor]:
  """
  Output:
  Args:
    splits: list of N 'row_splits' attribute of fid ragged tensors
    embeddings: list of N embeddings
    slice_dims: list of N slice_dims
  Output:
    reduced: M output tensors, and i-th tensor has a shape [bs, flat_slice_dims[i]].
    where flat_slice_dims=concat(slice_dims), and M=len(flat_slice_dims)
  """
  flat_slice_dims = []
  row_split_splits = []
  row_split_idx = 0
  for i in range(len(slice_dims)):
    s = slice_dims[i]
    flat_slice_dims.extend(s)

    row_split_splits.append(row_split_idx)
    row_split_idx += splits[i].shape[0]

  row_split_splits.append(row_split_idx)
  with tf.device("/device:CPU:0"):
    fused_splits = tf.cast(tf.concat(splits, 0), tf.int32)
  return gen_distribution_ops.monolith_fused_reduce_and_split_gpu(
      fused_splits,
      embeddings,
      slice_dims=flat_slice_dims,
      num_slices=len(flat_slice_dims),
      row_split_splits=row_split_splits,
      name=name)


@tf.RegisterGradient("MonolithFusedReduceAndSplitGPU")
def _fused_reduce_and_split_gpu_grad(op: tf.Operation, *grads):
  row_split_splits = op.get_attr('row_split_splits')
  slice_dims = op.get_attr('slice_dims')
  return [None] + gen_distribution_ops.monolith_fused_reduce_and_split_gpu_grad(
      op.inputs[0],
      op.inputs[1:len(row_split_splits)],
      grads,
      row_split_splits=row_split_splits,
      slice_dims=slice_dims)

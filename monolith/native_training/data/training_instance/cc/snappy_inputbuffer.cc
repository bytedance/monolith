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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <sys/time.h>
#include "glog/logging.h"

#include "monolith/native_training/data/training_instance/cc/snappy_inputbuffer.h"

namespace tensorflow {
namespace io {

ByteSnappyInputBuffer::ByteSnappyInputBuffer(
    RandomAccessFile* file,
    size_t input_buffer_bytes,  // size of input_buffer_
    size_t output_buffer_bytes  // size of output_buffer_
    )
    : file_(file),
      input_buffer_capacity_(input_buffer_bytes),
      output_buffer_capacity_(output_buffer_bytes),
      bytes_read_(0) {
  cached_mem_pool_ = CachedMemPool::init(input_buffer_bytes);
  input_buffer_ = cached_mem_pool_->allocate();
  output_buffer_ = cached_mem_pool_->allocate();
  next_in_ = input_buffer_.get();
  LOG_IF(ERROR, !ReadFromFile().ok()) << "Failed to read ahead from HDFS.";
}

ByteSnappyInputBuffer::~ByteSnappyInputBuffer() {
  cached_mem_pool_->deallocate(output_buffer_);
  cached_mem_pool_->deallocate(input_buffer_);
}

Status ByteSnappyInputBuffer::ReadNBytes(int64 bytes_to_read, tstring* result) {
  result->clear();
  result->resize_uninitialized(bytes_to_read);

  char* result_ptr = result->mdata();

  // Read as many bytes as possible from cache.
  size_t bytes_read = ReadBytesFromCache(bytes_to_read, result_ptr);
  bytes_to_read -= bytes_read;
  result_ptr += bytes_read;

  while (bytes_to_read > 0) {
    // Now that the cache is empty we need to inflate more data.
    TF_RETURN_IF_ERROR(Inflate());

    bytes_read = ReadBytesFromCache(bytes_to_read, result_ptr);
    bytes_to_read -= bytes_read;
    result_ptr += bytes_read;
  }

  return Status::OK();
}

int64 ByteSnappyInputBuffer::Tell() const { return bytes_read_; }

Status ByteSnappyInputBuffer::Reset() {
  file_pos_ = 0;
  avail_in_ = 0;
  avail_out_ = 0;
  next_in_ = input_buffer_.get();
  bytes_read_ = 0;
  return Status::OK();
}

size_t ByteSnappyInputBuffer::ReadBytesFromCache(size_t bytes_to_read,
                                                 char* result_ptr) {
  size_t can_read_bytes = std::min(bytes_to_read, avail_out_);
  if (can_read_bytes > 0) {
    memcpy(result_ptr, next_out_, can_read_bytes);
    next_out_ += can_read_bytes;
    avail_out_ -= can_read_bytes;
    bytes_read_ += can_read_bytes;
  }
  return can_read_bytes;
}

Status ByteSnappyInputBuffer::Inflate() {
  // Output buffer must have been cleared before uncompressing more input.
  DCHECK_EQ(avail_out_, 0);

  // Read origin length of a block.
  if (block_length_ == 0) {
    TF_RETURN_IF_ERROR(ReadBlockLength(&block_length_));

    // Output buffer must be large enough to fit the uncompressed block.
    DCHECK_GE(output_buffer_capacity_, block_length_);
  }

  // Read length of a compressed chunk.
  uint32 compressed_chunk_length = 0;
  TF_RETURN_IF_ERROR(ReadBlockLength(&compressed_chunk_length));

  // Read bytes to buffer a chunk
  if (avail_in_ < compressed_chunk_length) {
    TF_RETURN_IF_ERROR(ReadFromFile());
    if (avail_in_ < compressed_chunk_length) {
      if (compressed_chunk_length > input_buffer_capacity_) {
        // TODO(gaofei.gf): increase buffer size dynamically
        return errors::ResourceExhausted(
            "Input buffer(size: ", input_buffer_capacity_,
            " bytes) too small. Should be larger ", "than ",
            compressed_chunk_length, " bytes.");
      } else {
        return errors::OutOfRange("EOF reached with incomplete tail bytes.");
      }
    }
  }

  // Uncompress a chunk
  size_t chunk_length = 0;
  if (!port::Snappy_GetUncompressedLength(next_in_, compressed_chunk_length,
                                          &chunk_length)) {
    return errors::DataLoss("Snappy_GetUncompressedLength failed");
  }
  next_out_ = output_buffer_.get();
  if (!port::Snappy_Uncompress(next_in_, compressed_chunk_length, next_out_)) {
    return errors::DataLoss("Snappy_Uncompress failed");
  }
  next_in_ += compressed_chunk_length;
  avail_in_ -= compressed_chunk_length;
  avail_out_ += chunk_length;
  uncompressed_bytes_in_block_ += chunk_length;

  // Check a block is uncompressed
  if (uncompressed_bytes_in_block_ == block_length_) {
    block_length_ = 0;
    uncompressed_bytes_in_block_ = 0;
  }
  return Status::OK();
}

Status ByteSnappyInputBuffer::ReadBlockLength(uint32* length) {
  *length = 0;
  size_t bytes_to_read = 4;
  while (bytes_to_read > 0) {
    if (avail_in_ == 0) {
      TF_RETURN_IF_ERROR(ReadFromFile());
    }
    size_t readable = std::min(bytes_to_read, avail_in_);

    for (int i = 0; i < readable; i++) {
      // The "unsigned char" type cast is intentional to avoid implicit type
      // casting of the signed char to unsigned int during bitwise OR which
      // causes weird overflow errors.
      // Little endian
      *length = (*length << 8) | static_cast<unsigned char>(next_in_[0]);
      bytes_to_read--;
      next_in_++;
      avail_in_--;
    }
  }
  return Status::OK();
}

Status ByteSnappyInputBuffer::ReadFromFile() {
  int bytes_to_read = input_buffer_capacity_;
  char* read_offset = reinterpret_cast<char*>(input_buffer_.get());

  // If there are unread bytes in the input stream we move them to the head
  // of the stream to maximize the space available to read new data into.
  // TODO(srbs): A circular buffer would be useful here.
  if (avail_in_ > 0) {
    size_t read_bytes = next_in_ - input_buffer_.get();
    // Remove `read_bytes` from the head of the input stream.
    // Move unread bytes to the head of the input stream.
    if (read_bytes > 0) {
      memmove(input_buffer_.get(), next_in_, avail_in_);
    }

    bytes_to_read -= avail_in_;
    read_offset += avail_in_;
  }
  StringPiece data;
  // Try to read enough data to fill up input_buffer_.
  struct timeval t0;
  struct timeval t1;
  size_t old_size = data.size();
  gettimeofday(&t0, NULL);
  Status s = Status(error::OUT_OF_RANGE, "Read less bytes than requested");
  if (!reached_eof_) {
    read_round_++;
    s = file_->Read(file_pos_, bytes_to_read, &data, read_offset);
  }
  gettimeofday(&t1, NULL);
  int64_t elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
  elapsed /= 1000;
  auto throughput =
      (elapsed == 0) ? 0.0 : (data.size() - old_size) / (1024 * elapsed);
  LOG_EVERY_N(INFO, 100) << "********************************At round: "
                         << read_round_
                         << ", the expected read: " << bytes_to_read
                         << " and the actual read is:  "
                         << (data.size() - old_size) / (1024.0 * 1024)
                         << " MB at timestamp: " << elapsed
                         << " ms with a bandwidth: " << throughput
                         << " MBps. If out of range? "
                         << errors::IsOutOfRange(s)
                         << " .*********************************";
  if (data.data() != read_offset) {
    memmove(read_offset, data.data(), data.size());
  }

  // Since we moved unread data to the head of the input stream we can point
  // next_in to the head of the input stream.
  next_in_ = input_buffer_.get();

  // Note: data.size() could be different from bytes_to_read.
  avail_in_ += data.size();
  file_pos_ += data.size();

  // Report failure if not EoF or normal reading.
  if (!s.ok() && !errors::IsOutOfRange(s)) {
    return s;
  }

  // We throw OutOfRange error iff no new data has been read from file.
  // Since we never check how much data is remaining in the file, it is
  // possible that on the last read there isn't enough data in the file to
  // fill up the buffer in which case file_->ReadNBytes would return an
  // OutOfRange error.
  if (data.empty()) {
    reached_eof_ = true;
    return errors::OutOfRange("EOF reached");
  }
  if (errors::IsOutOfRange(s)) {
    reached_eof_ = true;
    return Status::OK();
  }

  return s;
}

}  // namespace io
}  // namespace tensorflow

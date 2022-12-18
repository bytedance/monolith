# Diff 1
Since we use shared library build version of TF, it is important to
use libtensorflow_framework.so.

However, by default it is not public, so we change the visibility here.

# Diff 2
To do the failure handling, we will check fail when we can't find the
remote device.

# Diff 3
Since in our training, we create multiple masters, this log info is too messy.

# Diff 4
Adds the ability to tune the default handler number.

# Diff 5
Add a macro protected Read over Pread to the HDFS random file accessor.
Read is preferred over Pread since it works better with caching and block
reader. Read improves the read performance for about 30%.

# Diff: GPU Support
In `tensorflow/core/kernels/BUILD`:
`gpu_device_array` is for array-of-tensors types used by our custom ops.
This patch exports `gpu_device_array` libs as target in a lighter manner. However if we builds on this original `gpu_device_array` [cc_lib](https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/core/kernels/BUILD#L725) by changing its visibility directly to public, it will fail to build because it duplicated-dependency on `//tensorflow/core:lib` which'd lead to the multiple definitions while building cuda.

# Diff 8:
There is no default timeout for RPCs like CleanGraph, which causes RPC hangs forever when the remote is down even if
grpc_fail_fast is true.

# Diff 9:
Temporarily solution for tensorflow deps.

# Diff 11:
Add tf_gpu_kernel_library_allow_except to compile cpu/gpu version hash table
# Diff 11 & 12:
Add allow_except rules to compile cpu/gpu version hash table

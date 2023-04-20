package controllers

const (
	ModuleInference      = "inference"
	MLPlatformVolcPrefix = "mlplatform.volcengine.com"
)

const (
	ImmutableLabelServiceId = ModuleInference + "." + MLPlatformVolcPrefix + "/service-id"
	ImmutableLabelRoleName  = ModuleInference + "." + MLPlatformVolcPrefix + "/role-name"
	ImmutableLabelShardId   = ModuleInference + "." + MLPlatformVolcPrefix + "/shard-id"
	ImmutableLabelShardNum  = ModuleInference + "." + MLPlatformVolcPrefix + "/shard-num"
)

const (
	EnvShardId     = "MLP_SHARD_ID"
	EnvPodName     = "MLP_POD_NAME"
	EnvHostIp      = "MLP_HOST_IP"
	EnvShardNum    = "MLP_SHARD_NUM"
	EnvIdc         = "MLP_IDC"
	EnvServiceName = "MLP_SERVICE_NAME"
	EnvRoleName    = "MLP_ROLE_NAME"
	EnvPort        = "MLP_%s_PORT"
)

// kubelet
const (
	PodInitializing   = "PodInitializing"
	ContainerCreating = "ContainerCreating"
)

const (
	DefaultRpcPort  = 8500
	DefaultHttpPort = 8501
)

const (
	ContainerEvicted = "Evicted"
)

// reason for pod
const (
	ReasonInsufficientClusterResources = "InsufficientClusterResources"
	ReasonInProgress                   = ""
	ReasonStatusNotFound               = "StatusNotFound"
	ReasonEvicted                      = "Evicted"
	ReasonServiceExceptionExited       = "ExceptionExited"
)

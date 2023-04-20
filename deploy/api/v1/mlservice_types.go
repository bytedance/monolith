/*
Copyright 2023.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1

import (
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// ServicePortType is the data type of ServicePort Type
type ServicePortType string

const (
	ServicePortTypeHttp    ServicePortType = "HTTP"
	ServicePortTypeRpc     ServicePortType = "RPC"
	ServicePortTypeMetrics ServicePortType = "Metrics"
	ServicePortTypeOther   ServicePortType = "Other"
)

// DeploymentTemplateSpec defines the metadata and spec of a Deployment
type DeploymentTemplateSpec struct {
	// Standard object metadata.
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// Specification of the desired behavior of the Deployment.
	Spec appsv1.DeploymentSpec `json:"spec"`
}

// ServicePort contains information on service's port.
type ServicePort struct {
	// The type of this port within the service.
	Type ServicePortType `json:"type,omitempty"`
	// The port that will be exposed by this service.
	Port int32 `json:"port"`
}

// ServiceSpec describes the attributes that a user creates on a service.
type ServiceSpec struct {
	// ServiceType defines which type of service need to be created
	ServiceType corev1.ServiceType `json:"serviceType,omitempty"`

	// The list of ports that are exposed by this service.
	// More info: https://kubernetes.io/docs/concepts/services-networking/service/#virtual-ips-and-service-proxies
	Ports []ServicePort `json:"ports,omitempty"`
}

// RoleSpec defines the desired state of a role in MLService
type RoleSpec struct {
	// Name of the role
	Name string `json:"name"`

	// Number of shards for the role, each shard associated with a Deployment
	ShardNum int32 `json:"shardNum,omitempty"`

	// Template of the DeploymentSpec
	Template DeploymentTemplateSpec `json:"template"`

	ServiceSpec *ServiceSpec `json:"serviceSpec,omitempty"`
}

// MLServiceSpec defines the desired state of MLService
type MLServiceSpec struct {
	// selector is a label query over deployment.
	// It must match the deployment template's labels.
	Selector *metav1.LabelSelector `json:"selector"`

	// Roles defines desired state for each role in the service
	Roles []RoleSpec `json:"roles"`
}

// ServicePhase is a label for the condition of a MLService at the current time.
type ServicePhase string

const (
	// ServiceQueuing means the service is queuing, waiting to be scheduled
	ServiceQueuing ServicePhase = "Queuing"
	// ServiceDeploying means pods of the service are scheduled and being initializing
	ServiceDeploying ServicePhase = "Deploying"
	// ServiceRunning means all pods of the service are running
	ServiceRunning ServicePhase = "Running"
	// ServiceAbnormal means some pods of the service are abnormal
	ServiceAbnormal ServicePhase = "Abnormal"
	// ServiceDeleting means the service is being deleted
	ServiceDeleting ServicePhase = "Deleting"
	// ServiceStopping means replicas of the service is being scaled down to 0
	ServiceStopping ServicePhase = "Stopping"
	// ServiceStopped means replicas of the service has been scaled down to 0
	ServiceStopped ServicePhase = "Stopped"
)

// MLServiceStatus defines the observed state of MLService
type MLServiceStatus struct {
	// Phase is a simple, high-level summary of where the Service is in its lifecycle.
	// +optional
	Phase ServicePhase `json:"phase,omitempty"`

	// RoleShardStatusMap shows the current status for all Deployments.
	// The key is Deployment name, value is its status info
	RoleShardStatusMap map[string]appsv1.DeploymentStatus `json:"roleShardStatusMap,omitempty"`

	// RoleShardStatusMap shows the current status for all Services.
	// The key is Service name, value is its status info
	RoleServiceStatusMap map[string]corev1.ServiceStatus `json:"roleServiceStatusMap,omitempty"`

	// RoleServiceClusterIps shows the cluster ip for all Services.
	// The key is Service name, value is its clusterIP
	RoleServiceClusterIps map[string]string `json:"roleServiceClusterIps,omitempty"`

	// LastTransitionTime is time the last Phase transitioned to current one.
	// +optional
	LastTransitionTime metav1.Time `json:"lastTransitionTime,omitempty"`

	// Unique, one-word, CamelCase reason for the phase's last transition.
	// +optional
	Reason string `json:"reason,omitempty"`

	// Human-readable message indicating details about last transition.
	// +optional
	Message string `json:"message,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:resource:path=mlservices,shortName=mlsvc

// MLService is the Schema for the mlservices API
type MLService struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   MLServiceSpec   `json:"spec,omitempty"`
	Status MLServiceStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// MLServiceList contains a list of MLService
type MLServiceList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []MLService `json:"items"`
}

func init() {
	SchemeBuilder.Register(&MLService{}, &MLServiceList{})
}

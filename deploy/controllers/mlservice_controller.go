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

package controllers

import (
	"context"
	"reflect"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	appsclient "k8s.io/client-go/kubernetes/typed/apps/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	monolithv1 "code.byted.org/data/monolith/deploy/api/v1"
)

type MLSvcHandler func(ctx context.Context, mlsvc *monolithv1.MLService) error

var handlers []MLSvcHandler

// MLServiceReconciler reconciles a MLService object
type MLServiceReconciler struct {
	appsclient.AppsV1Client
	client.Client
	Scheme *runtime.Scheme
}

//+kubebuilder:rbac:groups=mlplatform.volcengine.com,resources=mlservices,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=mlplatform.volcengine.com,resources=mlservices/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=mlplatform.volcengine.com,resources=mlservices/finalizers,verbs=update
//+kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=services,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=apps,resources=replicasets,verbs=get;list
//+kubebuilder:rbac:groups="",resources=pods,verbs=get;list

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the MLService object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.7.2/pkg/reconcile
func (r *MLServiceReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx).WithName("MLService").WithValues("mlservice", req.NamespacedName)

	var mlsvc monolithv1.MLService
	if err := r.Get(ctx, req.NamespacedName, &mlsvc); err != nil {
		log.Error(err, "unable to fetch MLService")
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	for _, h := range handlers {
		err := h(ctx, &mlsvc)
		if err != nil {
			log.Error(err, "handler failed")
			return ctrl.Result{}, err
		}
	}

	if err := r.updateStatus(ctx, &mlsvc); err != nil {
		log.Error(err, "unable to update status")
		return ctrl.Result{}, err
	}

	// if phase is Stopping, chances are that there will be no events to trigger the Reconcile,
	// so requeue is needed.
	if mlsvc.Status.Phase == monolithv1.ServiceStopping {
		return ctrl.Result{RequeueAfter: 2 * time.Second}, nil
	}

	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *MLServiceReconciler) SetupWithManager(mgr ctrl.Manager) error {
	handlers = []MLSvcHandler{
		r.DeploymentHandler,
		r.ServiceHandler,
	}

	return ctrl.NewControllerManagedBy(mgr).
		For(&monolithv1.MLService{}).
		Owns(&appsv1.Deployment{}).
		Owns(&corev1.Service{}).
		Complete(r)
}

// updateStatus update the status of MLService according to status of resources owned by this MLService
func (r *MLServiceReconciler) updateStatus(ctx context.Context, mlsvc *monolithv1.MLService) error {
	log := log.FromContext(ctx).WithName("MLService").WithValues("mlservice", mlsvc.Name)

	// List all deployment resources owned by this MLService
	deployments, err := r.getOwnedDeployments(ctx, mlsvc)
	if err != nil {
		log.Error(err, "get owned deployments failed")
		return err
	}

	// Shard status map
	var newRoleShardStatusMap map[string]appsv1.DeploymentStatus
	if len(deployments.Items) > 0 {
		newRoleShardStatusMap = make(map[string]appsv1.DeploymentStatus)
	}
	for _, dp := range deployments.Items {
		newRoleShardStatusMap[dp.Name] = *dp.Status.DeepCopy()
	}

	// List all service resources owned by this MLService
	services, err := r.getOwnedServices(ctx, mlsvc)
	if err != nil {
		log.Error(err, "get owned services failed")
		return err
	}

	// Service status map
	var newRoleServiceStatusMap map[string]corev1.ServiceStatus
	if len(services.Items) > 0 {
		newRoleServiceStatusMap = make(map[string]corev1.ServiceStatus)
	}
	for _, svc := range services.Items {
		newRoleServiceStatusMap[svc.Name] = *svc.Status.DeepCopy()
	}

	// Service ClusterIps
	var newRoleServiceClusterIps map[string]string
	if len(services.Items) > 0 {
		newRoleServiceClusterIps = make(map[string]string)
	}
	for _, svc := range services.Items {
		newRoleServiceClusterIps[svc.Name] = svc.Spec.ClusterIP
	}

	// phase, reason, message
	phase, reason, message, err := r.getMLServiceStatus(ctx, mlsvc)
	if err != nil {
		log.Error(err, "get MLService status failed")
		return err
	}

	if mlsvc.Status.Phase == phase && mlsvc.Status.Reason == reason && mlsvc.Status.Message == message &&
		reflect.DeepEqual(newRoleShardStatusMap, mlsvc.Status.RoleShardStatusMap) &&
		reflect.DeepEqual(newRoleServiceStatusMap, mlsvc.Status.RoleServiceStatusMap) &&
		reflect.DeepEqual(newRoleServiceClusterIps, mlsvc.Status.RoleServiceClusterIps) {
		log.Info("no changes of MLService status")
		return nil
	}

	// update MLService status
	log.Info("MLService status", "phase", phase, "reason", reason, "message", message)
	mlsvc.Status.RoleShardStatusMap = newRoleShardStatusMap
	mlsvc.Status.RoleServiceStatusMap = newRoleServiceStatusMap
	mlsvc.Status.RoleServiceClusterIps = newRoleServiceClusterIps
	mlsvc.Status.LastTransitionTime = metav1.Now()
	mlsvc.Status.Phase = phase
	mlsvc.Status.Reason = reason
	mlsvc.Status.Message = message
	if err := r.Status().Update(ctx, mlsvc); err != nil {
		log.Error(err, "unable to update MLService status")
		return err
	}

	return nil
}

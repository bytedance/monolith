package controllers

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	monolithv1 "code.byted.org/data/monolith/deploy/api/v1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// getDeploymentName returns Deployment name with pattern {mlsvcName}-{role}-{shardIdx}
func getDeploymentName(mlsvcName, role string, shardIdx int) string {
	return fmt.Sprintf("%s-%s-%d", mlsvcName, strings.ToLower(role), shardIdx)
}

// DeploymentHandler handles with k8s Deployment resource,
// make sure deployments owned by MLService in cluster match the desired state the MLService spec defines.
func (r *MLServiceReconciler) DeploymentHandler(ctx context.Context, mlsvc *monolithv1.MLService) error {
	if mlsvc == nil {
		return nil
	}

	log := log.FromContext(ctx).WithName("DeploymentHandler")

	// delete deployments if MLService is deleted
	mlsvcDeleting := !mlsvc.GetDeletionTimestamp().IsZero()
	if mlsvcDeleting {
		return r.cleanOwnedDeployments(ctx, mlsvc)
	}

	for roleIdx, role := range mlsvc.Spec.Roles {
		shardNum := int(role.ShardNum)
		if shardNum == 0 {
			// default value of ShardNum is 1
			shardNum = 1
		}

		for shardIdx := 1; shardIdx <= shardNum; shardIdx++ {
			deploy := &appsv1.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      getDeploymentName(mlsvc.Name, mlsvc.Spec.Roles[roleIdx].Name, shardIdx),
					Namespace: mlsvc.Namespace,
				},
			}

			if _, err := ctrl.CreateOrUpdate(ctx, r.Client, deploy, func() error {
				template := mlsvc.Spec.Roles[roleIdx].Template.DeepCopy()
				// set additional labels,annotations,label selector for deployment
				if template.ObjectMeta.Labels == nil {
					template.ObjectMeta.Labels = make(map[string]string, 0)
				}
				if template.ObjectMeta.Annotations == nil {
					template.ObjectMeta.Annotations = make(map[string]string, 0)
				}
				if template.Spec.Selector.MatchLabels == nil {
					template.Spec.Selector.MatchLabels = make(map[string]string, 0)
				}
				SetAdditionalKeyValuePairs(template.ObjectMeta.Labels, mlsvc.Name, role.Name, &shardIdx, &shardNum)
				SetAdditionalKeyValuePairs(template.ObjectMeta.Annotations, mlsvc.Name, role.Name, &shardIdx, &shardNum)
				SetAdditionalKeyValuePairs(template.Spec.Selector.MatchLabels, mlsvc.Name, role.Name, &shardIdx, &shardNum)

				// set additional labels for pod
				if template.Spec.Template.ObjectMeta.Labels == nil {
					template.Spec.Template.ObjectMeta.Labels = make(map[string]string, 0)
				}
				SetAdditionalKeyValuePairs(template.Spec.Template.ObjectMeta.Labels, mlsvc.Name, role.Name, &shardIdx, &shardNum)

				// set additional Env to the container
				idc, _ := mlsvc.GetAnnotations()[EnvIdc]
				var ports []corev1.ServicePort
				if mlsvc.Spec.Roles[roleIdx].ServiceSpec != nil {
					ports = GetServicePorts(mlsvc.Spec.Roles[roleIdx].ServiceSpec.Ports)
				}
				for i := range template.Spec.Template.Spec.Containers {
					template.Spec.Template.Spec.Containers[i].Env = append(template.Spec.Template.Spec.Containers[i].Env,
						AdditionalEnvs(mlsvc.Name, role.Name, idc, shardIdx, int(shardNum), ports)...,
					)
				}

				deploy.ResourceVersion = ""

				// set ObjectMeta.Labels
				if deploy.ObjectMeta.Labels == nil {
					deploy.ObjectMeta.Labels = make(map[string]string)
				}
				for k, v := range template.ObjectMeta.Labels {
					deploy.ObjectMeta.Labels[k] = v
				}

				// set ObjectMeta.Annotations
				if deploy.ObjectMeta.Annotations == nil {
					deploy.ObjectMeta.Annotations = make(map[string]string)
				}
				for k, v := range template.ObjectMeta.Annotations {
					deploy.ObjectMeta.Annotations[k] = v
				}

				// set Finalizers
				for _, finalizer := range template.ObjectMeta.Finalizers {
					controllerutil.AddFinalizer(deploy, finalizer)
				}

				// set Spec
				deploy.Spec = template.Spec

				// set the owner so that garbage collection can kicks in
				if err := ctrl.SetControllerReference(mlsvc, deploy, r.Scheme); err != nil {
					log.Error(err, "unable to set ownerReference from MLService to Deployment")
					return err
				}

				// end of ctrl.CreateOrUpdate
				return nil
			}); err != nil {
				// error handling of ctrl.CreateOrUpdate
				log.Error(err, "unable to ensure deployment is correct")
				return err
			}
		}
	}

	return nil
}

func (r *MLServiceReconciler) createDeployment(ctx context.Context, dp *appsv1.Deployment) error {
	log := log.FromContext(ctx).WithValues("DeploymentName", dp.Name)
	if err := r.Client.Create(ctx, dp); err != nil {
		log.Error(err, "failed to create Deployment resource")
		return err
	}

	log.Info("created Deployment resource for MLService")
	return nil
}

func (r *MLServiceReconciler) updateDeployment(ctx context.Context, desired, existing *appsv1.Deployment) error {
	log := log.FromContext(ctx).WithValues("DeploymentName", existing.Name)
	if equality.Semantic.DeepEqual(existing, desired) {
		return nil
	}

	if err := r.Client.Update(ctx, desired); err != nil {
		log.Error(err, "failed to update Deployment resource")
		return err
	}
	log.Info("update Deployment resource for MLService")
	return nil
}

func (r *MLServiceReconciler) deleteDeployment(ctx context.Context, dp *appsv1.Deployment) error {
	log := log.FromContext(ctx).WithValues("DeploymentName", dp.Name)
	if err := r.Client.Delete(ctx, dp); err != nil {
		log.Error(err, "failed to delete Deployment resource")
		return err
	}

	log.Info("delete deployment resource: " + dp.Name)
	return nil
}

// cleanOwnedDeployments will delete any existing Deployment resources that
// were created for the given MLService
func (r *MLServiceReconciler) cleanOwnedDeployments(ctx context.Context, mlsvc *monolithv1.MLService) error {
	log := log.FromContext(ctx).WithValues("MLService", mlsvc.Name)
	log.Info("finding existing Deployments for MLService resource")

	// list all deployment resources owned by this MLService
	deployments, err := r.getOwnedDeployments(ctx, mlsvc)
	if err != nil {
		return err
	}

	for _, deployment := range deployments.Items {
		if !deployment.GetDeletionTimestamp().IsZero() {
			// deployment already deleted, ignore.
			continue
		}

		// delete deployment
		if err := r.Delete(ctx, &deployment); err != nil {
			log.Error(err, "failed to delete Deployment resource: "+deployment.Name)
			return err
		}

		log.Info("delete deployment resource: " + deployment.Name)
	}

	return nil
}

// getOwnedDeployments return all deployments owned by the MLService
func (r *MLServiceReconciler) getOwnedDeployments(ctx context.Context, mlsvc *monolithv1.MLService) (*appsv1.DeploymentList, error) {
	var deployments appsv1.DeploymentList
	if err := r.List(ctx, &deployments, client.InNamespace(mlsvc.Namespace),
		client.MatchingLabels(mlsvc.Spec.Selector.MatchLabels)); err != nil {
		return nil, err
	}
	return &deployments, nil
}

// AdditionalEnvs return a list of EnvVar, these Envs will be injected to pod container
func AdditionalEnvs(mlsvcName, roleName, idc string, shardIdx, shardNum int, ports []corev1.ServicePort) []corev1.EnvVar {
	envs := []corev1.EnvVar{
		{
			Name:  EnvShardId,
			Value: strconv.Itoa(shardIdx),
		},
		{
			Name:  EnvShardNum,
			Value: strconv.Itoa(int(shardNum)),
		},
		{
			Name:  EnvServiceName,
			Value: mlsvcName,
		},
		{
			Name:  EnvRoleName,
			Value: roleName,
		},
		{
			Name:  EnvIdc,
			Value: idc,
		},
		{
			Name: EnvPodName,
			ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{
					FieldPath: "metadata.name",
				},
			},
		},
		{
			Name: EnvHostIp,
			ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{
					FieldPath: "status.podIP",
				},
			},
		},
	}
	for _, port := range ports {
		envs = append(envs, corev1.EnvVar{
			Name:  fmt.Sprintf(EnvPort, strings.ToUpper(string(port.Name))),
			Value: strconv.Itoa(int(port.Port)),
		})
	}
	return envs
}

// SetAdditionalKeyValuePairs inserts additional labels to the existing Labels map
func SetAdditionalKeyValuePairs(existing map[string]string, mlsvcName, roleName string, shardIdx, shardNum *int) {
	additional := map[string]string{
		ImmutableLabelServiceId: mlsvcName,
		ImmutableLabelRoleName:  roleName,
	}

	if shardIdx != nil {
		additional[ImmutableLabelShardId] = strconv.Itoa(*shardIdx)
	}

	if shardNum != nil {
		additional[ImmutableLabelShardNum] = strconv.Itoa(*shardNum)
	}

	for k, v := range additional {
		existing[k] = v
	}
}

package controllers

import (
	"context"
	"errors"
	"fmt"
	"strings"

	monolithv1 "code.byted.org/data/monolith/deploy/api/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// getServiceName returns Service name with pattern {mlsvcName}-{role}
func getServiceName(mlsvcName, role string) string {
	return fmt.Sprintf("%s-%s", mlsvcName, strings.ToLower(role))
}

// ServiceHandler handles with k8s Service resource,
// make sure k8s service owned by MLService in cluster match the desired state the MLService spec defines.
func (r *MLServiceReconciler) ServiceHandler(ctx context.Context, mlsvc *monolithv1.MLService) error {
	if mlsvc == nil {
		return nil
	}

	log := log.FromContext(ctx).WithName("ServiceHandler")

	// delete sesrvice if MLService is deleted
	mlsvcDeleting := !mlsvc.GetDeletionTimestamp().IsZero()
	if mlsvcDeleting {
		return r.cleanOwnedServices(ctx, mlsvc)
	}

	for roleIdx, role := range mlsvc.Spec.Roles {
		if role.ServiceSpec == nil {
			continue
		}

		if role.ServiceSpec.ServiceType != corev1.ServiceTypeClusterIP {
			mlsvc.Status.Phase = monolithv1.ServiceAbnormal
			mlsvc.Status.Message = "Currently only ClusterIP type is supported"
			log.Info("invalid service type, set status to abnormal", "ServiceType", role.ServiceSpec.ServiceType)
			if err := r.Status().Update(ctx, mlsvc); err != nil {
				log.Error(err, "unable to update MLService status")
				return err
			}
			return errors.New(mlsvc.Status.Message)
		}

		svc := &corev1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name:      getServiceName(mlsvc.Name, mlsvc.Spec.Roles[roleIdx].Name),
				Namespace: mlsvc.Namespace,
			},
		}

		if _, err := ctrl.CreateOrUpdate(ctx, r.Client, svc, func() error {
			svc.Spec = corev1.ServiceSpec{
				Ports:    GetServicePorts(role.ServiceSpec.Ports),
				Selector: map[string]string{},
				Type:     corev1.ServiceTypeClusterIP,
			}

			// set service labels
			svc.ObjectMeta.Labels = make(map[string]string, 0)
			SetAdditionalKeyValuePairs(svc.ObjectMeta.Labels, mlsvc.Name, role.Name, nil, nil)
			for k, v := range mlsvc.Spec.Selector.MatchLabels {
				svc.ObjectMeta.Labels[k] = v
			}

			// set selector for pods
			SetAdditionalKeyValuePairs(svc.Spec.Selector, mlsvc.Name, role.Name, nil, nil)

			// set the owner so that garbage collection can kicks in
			if err := ctrl.SetControllerReference(mlsvc, svc, r.Scheme); err != nil {
				log.Error(err, "unable to set ownerReference from MLService to Service")
				return err
			}

			// end of ctrl.CreateOrUpdate
			return nil
		}); err != nil {
			// error handling of ctrl.CreateOrUpdate
			log.Error(err, "unable to ensure service is correct")
			return err
		}

	}
	return nil
}

// cleanOwnedServices will delete any existing Service resources that
// were created for the given MLService
func (r *MLServiceReconciler) cleanOwnedServices(ctx context.Context, mlsvc *monolithv1.MLService) error {
	log := log.FromContext(ctx).WithValues("MLService", mlsvc.Name)
	log.Info("finding existing Service for MLService resource")

	// List all service resources owned by this MLService
	services, err := r.getOwnedServices(ctx, mlsvc)
	if err != nil {
		return err
	}

	for _, svc := range services.Items {
		if !svc.GetDeletionTimestamp().IsZero() {
			// Service already deleted, ignore.
			continue
		}

		// Delete service
		if err := r.Delete(ctx, &svc); err != nil {
			log.Error(err, "failed to delete Service resource: "+svc.Name)
			return err
		}

		log.Info("delete service resource: " + svc.Name)
	}

	return nil
}

// getOwnedServices return all services owned by the MLService
func (r *MLServiceReconciler) getOwnedServices(ctx context.Context, mlsvc *monolithv1.MLService) (*corev1.ServiceList, error) {
	var services corev1.ServiceList
	if err := r.List(ctx, &services, client.InNamespace(mlsvc.Namespace),
		client.MatchingLabels(mlsvc.Spec.Selector.MatchLabels)); err != nil {
		return nil, err
	}
	return &services, nil
}

// GetServicePorts return HTTP port and gRPC port
func GetServicePorts(ports []monolithv1.ServicePort) []corev1.ServicePort {
	var httpPort int32 = DefaultHttpPort
	var rpcPort int32 = DefaultRpcPort
	for _, port := range ports {
		if port.Type == monolithv1.ServicePortTypeHttp {
			httpPort = port.Port
		} else if port.Type == monolithv1.ServicePortTypeRpc {
			rpcPort = port.Port
		} else {
			// ignore non-http and non-rpc port
			continue
		}
	}
	return []corev1.ServicePort{
		{
			Name: strings.ToLower(string(monolithv1.ServicePortTypeHttp)),
			Port: httpPort,
			TargetPort: intstr.IntOrString{
				Type:   intstr.Int,
				IntVal: int32(httpPort),
			},
		},
		{
			Name: strings.ToLower(string(monolithv1.ServicePortTypeRpc)),
			Port: rpcPort,
			TargetPort: intstr.IntOrString{
				Type:   intstr.Int,
				IntVal: int32(rpcPort),
			},
		},
	}
}

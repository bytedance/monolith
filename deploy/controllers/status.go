package controllers

import (
	"context"
	"fmt"

	monolithv1 "code.byted.org/data/monolith/deploy/api/v1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	utildeployment "k8s.io/kubectl/pkg/util/deployment"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// getMLServiceStatus return the status of MLService along with a reason and message.
// Queuing: at least one deployment owned by this MLService is in status Queuing
// Deploying: at least one deployment owned by this MLService is in status Deploying
// Running: all deployments owned by this MLService is in status Running
// Abnormal: at least one deployment owned by this MLService is in status Abnormal
// Deleting: MLService DeletionTimestamp is not zero
// Stopping: at least one deployment owned by this MLService is in status Stopping
// Stopped: all deployments owned by this MLService is in status Stopped
func (r *MLServiceReconciler) getMLServiceStatus(ctx context.Context, mlsvc *monolithv1.MLService) (phase monolithv1.ServicePhase, reason, message string, err error) {
	log := log.FromContext(ctx).WithName("MLService").WithValues("mlservice", mlsvc.Name)
	// the status of Deleting
	if !mlsvc.GetDeletionTimestamp().IsZero() {
		phase = monolithv1.ServiceDeleting
		return
	}

	// List all deployment resources owned by this MLService
	deployments, err := r.getOwnedDeployments(ctx, mlsvc)
	if err != nil {
		log.Error(err, "get owned deployments failed")
		return
	}

	deploymentStatusCount := make(map[monolithv1.ServicePhase]int, 0)
	for _, deployment := range deployments.Items {
		deploymentPhase, deploymentReason, deploymentMessage, dErr := r.getDeploymentStatus(ctx, &deployment)
		if dErr != nil {
			err = dErr
			log.Error(err, "get status of deployment failed", "deployment", deployment.Name)
			return
		}

		deploymentStatusCount[deploymentPhase]++

		// at least one is abnormal
		if deploymentPhase == monolithv1.ServiceAbnormal {
			phase = deploymentPhase
			reason = deploymentReason
			if deploymentMessage != "" {
				message = fmt.Sprintf("[Deployment %s] %s", deployment.Name, deploymentMessage)
			}
			log.Info("at least one deployment is abnormal", "deployment", deployment.Name, "phase", phase, "reason", reason, "message", message)
			return
		}

		// at least one is queuing
		if deploymentPhase == monolithv1.ServiceQueuing {
			phase = deploymentPhase
			log.Info("at least one deployment is queuing", "deployment", deployment.Name, "phase", phase)
			return
		}

		// at least one is stopping
		if deploymentPhase == monolithv1.ServiceStopping {
			phase = deploymentPhase
			log.Info("at least one deployment is stopping", "deployment", deployment.Name, "phase", phase)
			return
		}
	}

	// all deployments Running
	if count, ok := deploymentStatusCount[monolithv1.ServiceRunning]; ok && count == len(deployments.Items) {
		phase = monolithv1.ServiceRunning
		log.Info("all deployments are running", "phase", phase)
		return
	}

	// all deployments Stopped
	if count, ok := deploymentStatusCount[monolithv1.ServiceStopped]; ok && count == len(deployments.Items) {
		phase = monolithv1.ServiceStopped
		log.Info("all deployments are stopped", "phase", phase)
		return
	}

	phase = monolithv1.ServiceDeploying
	return
}

// getDeploymentStatus return the status of Deployment along with a reason and message.
// status is generated based on the latest replicaset
// Queuing: the latest ReplicaSet not exists, or it's status is Queuing
// Deploying: status of the latest ReplicaSet is Deploying
// Running: status of the latest ReplicaSet is Running
// Abnormal: status of the latest ReplicaSet is Abnormal
// Deleting: Deployment  DeletionTimestamp is not zero
// Stopping: the latest ReplicaSet is Stopping
// Stopped: the latest ReplicaSet is Stopped
func (r *MLServiceReconciler) getDeploymentStatus(ctx context.Context, deployment *appsv1.Deployment) (phase monolithv1.ServicePhase, reason, message string, err error) {
	log := log.FromContext(ctx).WithName("Deployment").WithValues("deployment", deployment.Name)

	// the status of Deleting
	if !deployment.GetDeletionTimestamp().IsZero() {
		phase = monolithv1.ServiceDeleting
		return
	}

	// get all replicaset
	var replicasets appsv1.ReplicaSetList
	if err = r.List(ctx, &replicasets, client.InNamespace(deployment.Namespace),
		client.MatchingLabels(deployment.Spec.Selector.MatchLabels)); err != nil {
		log.Error(err, "list replicasets failed")
		return
	}

	// get latest replicaset
	_, _, latest, err := utildeployment.GetAllReplicaSets(deployment, &r.AppsV1Client)
	if latest == nil {
		log.Info("latest replicaset not found, set phase to queuing")
		phase = monolithv1.ServiceQueuing
		return
	}

	return r.getReplicaSetStatus(ctx, latest)
}

// getReplicaSetStatus return the status of ReplicaSet along with a reason and message.
// Queuing: 1)  All Pods are in status Queuing 2)PodGroup is Pending;
// Deploying: at least one Pod is in status Deploying
// Running:  at least one Pod is in status Running
// Abnormal: all Pods are in status Abnormal
// Deleting: ReplicaSet DeletionTimestamp is not zero
// Stopping: replicas is 0 but pods exits
// Stopped: replicas is 0 and no pods exits
func (r *MLServiceReconciler) getReplicaSetStatus(ctx context.Context, replicaset *appsv1.ReplicaSet) (phase monolithv1.ServicePhase, reason, message string, err error) {
	log := log.FromContext(ctx).WithName("ReplicaSet").WithValues("replicaset", replicaset.Name)

	// list all pods of the replicaset
	var podList corev1.PodList
	if err = r.List(ctx, &podList, client.InNamespace(replicaset.Namespace),
		client.MatchingLabels(replicaset.Spec.Selector.MatchLabels)); client.IgnoreNotFound(err) != nil {
		log.Error(err, "list pods failed")
		return
	}

	// Stopping
	if *replicaset.Spec.Replicas == 0 && len(podList.Items) != 0 {
		phase = monolithv1.ServiceStopping
		return
	}

	// Stopped
	if *replicaset.Spec.Replicas == 0 && len(podList.Items) == 0 {
		phase = monolithv1.ServiceStopped
		return
	}

	podStatusCount := make(map[monolithv1.ServicePhase]int, 0)
	var abnormalReason, abnormalMessage string
	for _, pod := range podList.Items {
		podPhase, podReason, podMessage, dErr := r.getPodStatus(ctx, &pod)
		if dErr != nil {
			err = dErr
			log.Error(err, "get pod status failed")
			return
		}

		log.Info("pod status", "pod", pod.Name, "phase", podPhase, "reason", podReason, "message", podMessage)
		podStatusCount[podPhase]++

		// at least one pod is deploying or running
		if podPhase == monolithv1.ServiceDeploying || podPhase == monolithv1.ServiceRunning {
			phase = podPhase
			reason = podReason
			message = fmt.Sprintf("[Pod %s] %s", pod.Name, podMessage)
			log.Info(fmt.Sprintf("at least one pod is %s", phase), "pod", pod.Name)
			return
		}

		if podPhase == monolithv1.ServiceAbnormal && (podReason != "" || podMessage != "") {
			abnormalReason = podReason
			abnormalMessage = podMessage
		}
	}

	// all pods Queuing
	if count, ok := podStatusCount[monolithv1.ServiceQueuing]; ok && count == len(podList.Items) {
		log.Info("all pods are queuing")
		phase = monolithv1.ServiceQueuing
		return
	}

	// all pods Abnormal
	if count, ok := podStatusCount[monolithv1.ServiceAbnormal]; ok && count == len(podList.Items) {
		log.Info("all pods are abnormal")
		phase = monolithv1.ServiceAbnormal
		reason = abnormalReason
		message = abnormalMessage
		return
	}

	return
}

// getPodStatus return the status of Pod along with a reason and message.
// Queuing: condition PodScheduled is False
// Deploying: condition PodScheduled is True
// Running:  condition Ready is True
// Abnormal: Pod phase Succeeded、Failed、Unknown, Pending or Running but crash
// Deleting: ReplicaSet  DeletionTimestamp is not zero
// ref: https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-phase
func (r *MLServiceReconciler) getPodStatus(ctx context.Context, pod *corev1.Pod) (phase monolithv1.ServicePhase, reason, message string, err error) {
	log := log.FromContext(ctx).WithName("Pod").WithValues("pod", pod.Name)

	// the status of Deleting
	if !pod.GetDeletionTimestamp().IsZero() {
		phase = monolithv1.ServiceDeleting
		return
	}

	// Queuing
	if cond := getPodCondition(pod, corev1.PodScheduled); cond != nil && cond.Status != corev1.ConditionTrue {
		log.Info("PodScheduled condition is false, set phase to queuing")
		phase = monolithv1.ServiceQueuing
		if cond.Message == "" {
			reason = ReasonInProgress
		} else {
			reason = ReasonInsufficientClusterResources
		}
		return
	}

	// Running
	if cond := getPodCondition(pod, corev1.PodReady); cond != nil && cond.Status == corev1.ConditionTrue {
		log.Info("PodReady condition is true, set phase to running")
		phase = monolithv1.ServiceRunning
		return
	}

	// Abnormal
	// Abnormal case 1: pod Failure
	if pod.Status.Phase == corev1.PodFailed || pod.Status.Phase == corev1.PodSucceeded {
		log.Info(fmt.Sprintf("pod Phase is %s, set phase to abnormal", pod.Status.Phase))
		phase = monolithv1.ServiceAbnormal
		if pod.Status.Reason == ContainerEvicted {
			reason = ReasonEvicted
			message = "pod evicted"
		} else {
			reason = ReasonServiceExceptionExited
			message = "pod exited unexpectedly"
		}
		return
	}
	// Abnormal case 2: pod status unknown
	if pod.Status.Phase == corev1.PodUnknown {
		log.Info("pod Phase is Unknown, set phase to abnormal")
		phase = monolithv1.ServiceAbnormal
		reason = ReasonStatusNotFound
		message = "pod in status Unknown"
		return
	}
	// Abnormal case 3: container creating error or exited
	if pod.Status.Phase == corev1.PodRunning || pod.Status.Phase == corev1.PodPending {
		for _, status := range pod.Status.InitContainerStatuses {
			if tmpReason, tmpMessage := getContainerAbnormalMessage(status, true); tmpReason != "" {
				phase = monolithv1.ServiceAbnormal
				reason = tmpReason
				message = tmpMessage
				log.Info(fmt.Sprintf("pod Phase %s, but InitContainer is abnormal", pod.Status.Phase), "reason", reason, "message", message)
				return
			}
		}
		for _, status := range pod.Status.ContainerStatuses {
			if tmpReason, tmpMessage := getContainerAbnormalMessage(status, false); tmpReason != "" {
				phase = monolithv1.ServiceAbnormal
				reason = tmpReason
				message = tmpMessage
				log.Info(fmt.Sprintf("pod Phase %s, but container is abnormal", pod.Status.Phase), "reason", reason, "message", message)

				return
			}
		}
	}

	log.Info("assume phase is deploying in other cases")
	phase = monolithv1.ServiceDeploying
	return
}

func getContainerAbnormalMessage(status corev1.ContainerStatus, isInitContainer bool) (reason, message string) {
	waiting, terminated := status.State.Waiting, status.State.Terminated
	if waiting != nil && waiting.Reason != PodInitializing && waiting.Reason != ContainerCreating {
		return waiting.Reason, waiting.Message
	}
	if terminated != nil {
		if isInitContainer && terminated.ExitCode != 0 {
			reason = terminated.Reason
			message = terminated.Message
		}

		if !isInitContainer {
			reason = terminated.Reason
			if terminated.Message == "" {
				message = "Container terminated."
			} else {
				message = terminated.Message
			}
		}
		return
	}
	return
}

func getPodCondition(pod *corev1.Pod, conditionType corev1.PodConditionType) *corev1.PodCondition {
	for _, cond := range pod.Status.Conditions {
		if cond.Type == conditionType {
			return &cond
		}
	}
	return nil
}

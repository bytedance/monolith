# Distributed async training on EKS

To scale to multiple machines and handle failure recovery, we can utilize container orchestration frameworks such as yarn and kubernetes. Regradless what tool you use, as long as the `TF_CONFIG` environment variable is correctly set for each worker and ps, it will work just fine. 

In this tutorial, we will show how to setup distributed training using kubernetes, kubeflow, and AWS's elastic kubernetes service (EKS). Kubeflow is used as the middleware that injects `TF_CONFIG` environment variable for each worker container. 

## Prerequisite

Setup kubeflow on AWS by following the official guide. It will also help you to setup other tools such as aws cli and eksctl. Make sure to complete 

- Prerequisites
- Create an EKS Cluster
- Vanilla Installation

https://awslabs.github.io/kubeflow-manifests/docs/deployment/


## Prepare monolith docker

TODO

## Write Spec and launch training

If you have completed all the prerequisites, `kubectl` should be able to connect to your cluster on AWS. 

Now, create a spec file called `aws-tfjob.yaml`. 

```yaml
apiVersion: "kubeflow.org/v1"
kind: "TFJob"
metadata:
  name: "monolith-train"
  namespace: kubeflow 
spec:
  runPolicy:
    cleanPodPolicy: None
  tfReplicaSpecs:
    Worker:
      replicas: 4
      restartPolicy: Never
      template:
        metadata:
          annotations:
            # solve RBAC permission problem
            sidecar.istio.io/inject: "false"
        spec:
          containers:
            - name: tensorflow
              image: YOUR_IMAGE
              args: 
                - --model_dir=/tmp/model
    PS:
      replicas: 4
      restartPolicy: Never
      template:
        metadata:
          annotations:
            sidecar.istio.io/inject: "false"
        spec:
          containers:
            - name: tensorflow
              image: YOUR_IMAGE
              args:
                - --model_dir=/tmp/model
```

Then, launch training: 

```bash
kubectl apply -f aws-tfjob.yaml
```

To view the status of workers, you can use

```bash
# use this to list pods
kubectl --namespace kubeflow get pods
# use this get a log of a worker
kubectl --namespace kubeflow logs monolith-train-worker-0
```

Of course, there are other middlewares built on top of kubeflow to better help you to keep track of the training progress. Monolith's compatibility with tensorflow means that tools that are built for tensorflow will likely work with Monolith too. 
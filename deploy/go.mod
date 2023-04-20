module code.byted.org/data/monolith/deploy

go 1.15

require (
	github.com/onsi/ginkgo v1.16.5 // indirect
	github.com/onsi/gomega v1.18.1 // indirect
	k8s.io/api v0.23.5
	k8s.io/apimachinery v0.23.5
	k8s.io/client-go v0.23.5
	k8s.io/kubectl v0.20.6
	sigs.k8s.io/controller-runtime v0.10.2

)

replace (
	sigs.k8s.io/controller-runtime => sigs.k8s.io/controller-runtime v0.8.3
	k8s.io/api => k8s.io/api v0.20.6
	k8s.io/apimachinery => k8s.io/apimachinery v0.20.6
	k8s.io/client-go => k8s.io/client-go v0.20.6
)
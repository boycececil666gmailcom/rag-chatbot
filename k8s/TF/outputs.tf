output "namespace_name" {
  description = "Name of the created Kubernetes namespace"
  value       = kubectl_manifest.namespace.name
}

output "managed_resources" {
  description = "List of managed Kubernetes resources"
  value = [
    kubectl_manifest.namespace.name,
    kubectl_manifest.backend_deployment.name,
    kubectl_manifest.gateway_deployment.name,
    kubectl_manifest.qdrant_statefulset.name,
    kubectl_manifest.ingress.name
  ]
}

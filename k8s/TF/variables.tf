variable "kubeconfig_path" {
  description = "Path to the kubeconfig file"
  type        = string
  default     = "~/.kube/config"
}

variable "kube_context" {
  description = "Kubernetes context to use"
  type        = string
  default     = null
}

variable "namespace" {
  description = "Kubernetes namespace to deploy into"
  type        = string
  default     = "theme-based-rag-workflow-jiragraph"
}

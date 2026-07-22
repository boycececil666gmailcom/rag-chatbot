provider "kubectl" {
  config_path    = pathexpand(var.kubeconfig_path)
  config_context = var.kube_context
}

provider "kubernetes" {
  config_path    = pathexpand(var.kubeconfig_path)
  config_context = var.kube_context
}

# 1. Namespace Resource
resource "kubectl_manifest" "namespace" {
  yaml_body = file("${path.module}/../namespace.yaml")
}

# 2. Secrets Resource (Applied if secrets.yaml exists)
resource "kubectl_manifest" "secrets" {
  count      = fileexists("${path.module}/../secrets.yaml") ? 1 : 0
  yaml_body  = file("${path.module}/../secrets.yaml")
  depends_on = [kubectl_manifest.namespace]
}

# 3. Workload Resources (Backend, Gateway, Qdrant)
resource "kubectl_manifest" "backend_deployment" {
  yaml_body  = file("${path.module}/../backend-deployment.yaml")
  depends_on = [kubectl_manifest.namespace, kubectl_manifest.secrets]
}

resource "kubectl_manifest" "gateway_deployment" {
  yaml_body  = file("${path.module}/../gateway-deployment.yaml")
  depends_on = [kubectl_manifest.namespace]
}

resource "kubectl_manifest" "qdrant_statefulset" {
  yaml_body  = file("${path.module}/../qdrant-statefulset.yaml")
  depends_on = [kubectl_manifest.namespace]
}

# 4. Ingress Resource
resource "kubectl_manifest" "ingress" {
  yaml_body  = file("${path.module}/../ingress.yaml")
  depends_on = [kubectl_manifest.backend_deployment, kubectl_manifest.gateway_deployment]
}

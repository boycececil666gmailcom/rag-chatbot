# ==============================================================================
# Generate Kubernetes YAML Manifests from Terraform HCL using yamlencode
# ==============================================================================

# 1. Generate Namespace YAML
resource "local_file" "generated_namespace_yaml" {
  filename = "${path.module}/generated_namespace.yaml"
  content  = yamlencode({
    apiVersion = "v1"
    kind       = "Namespace"
    metadata = {
      name = var.namespace
      labels = {
        "app.kubernetes.io/name" = var.namespace
        environment              = "production"
      }
    }
  })
}

# 2. Generate Backend Service & Deployment YAML
resource "local_file" "generated_backend_yaml" {
  filename = "${path.module}/generated_backend.yaml"
  content  = format("%s---\n%s",
    yamlencode({
      apiVersion = "v1"
      kind       = "Service"
      metadata = {
        name      = "theme-based-rag-backend-service"
        namespace = var.namespace
        labels    = { app = "theme-based-rag-backend" }
      }
      spec = {
        type  = "ClusterIP"
        ports = [{ port = 80, targetPort = 8000, name = "http" }]
        selector = { app = "theme-based-rag-backend" }
      }
    }),
    yamlencode({
      apiVersion = "apps/v1"
      kind       = "Deployment"
      metadata = {
        name      = "theme-based-rag-backend"
        namespace = var.namespace
        labels    = { app = "theme-based-rag-backend" }
      }
      spec = {
        replicas = 2
        selector = { matchLabels = { app = "theme-based-rag-backend" } }
        template = {
          metadata = { labels = { app = "theme-based-rag-backend" } }
          spec = {
            containers = [
              {
                name            = "theme-based-rag-backend"
                image           = "theme-based-rag-backend:latest"
                imagePullPolicy = "Always"
                ports           = [{ containerPort = 8000, name = "http" }]
                env = [
                  { name = "QDRANT_URL", value = "http://qdrant-service:6333" },
                  { name = "GEMINI_MODEL", value = "gemini-3.1-flash-lite" },
                  { name = "GEMINI_EMBED_MODEL", value = "gemini-embedding-001" },
                  { name = "GEMINI_TEMPERATURE", value = "0.0" },
                  { name = "CHATBOT_THEME", value = "Fintech SaaS platform" }
                ]
              }
            ]
          }
        }
      }
    })
  )
}

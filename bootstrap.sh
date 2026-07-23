#!/bin/bash

# Define log_step function for beautiful output
log_step() {
    echo -e "\n\033[1;96m========================================================\033[0m"
    echo -e "\033[1;92m>>> $1 [$(basename "$0")] $2\033[0m"
    echo -e "\033[1;96m========================================================\033[0m\n"
}

# Step 1: Initialize local python environment
log_step "[1/4]" "Setting Up Python Virtual Environment & Dependencies"
source ./setup_env.sh

# Step 2: Build Docker images locally
log_step "[2/4]" "Building Local Docker Images"
docker info >/dev/null 2>&1 || { echo -e "\033[1;31mError: Docker daemon is not running. Please start Docker and try again.\033[0m"; exit 1; }

echo "Building core chatbot backend image..."
docker build -t theme-based-rag-backend:latest -f src/theme_based_rag_backend/Dockerfile . || exit 1

echo "Building API gateway image..."
docker build -t theme-based-rag-gateway:latest -f src/theme_based_rag_gateway/Dockerfile . || exit 1

# Step 3: Deploy manifests to Kubernetes
log_step "[3/4]" "Deploying to Kubernetes Cluster"

if [ ! -f "k8s/secrets.yaml" ]; then
    echo -e "\033[1;31mError: k8s/secrets.yaml is missing!\033[0m"
    echo "Please create k8s/secrets.yaml from k8s/secrets.yaml.template with your base64 encoded GEMINI_API_KEY first."
    exit 1
fi

echo "Applying Kubernetes manifests via Terraform..."
pushd k8s/TF >/dev/null || exit 1
terraform init || exit 1
terraform apply -auto-approve || exit 1
popd >/dev/null || exit 1

# Step 4: Access Guide (Pure Ingress Mode)
log_step "[4/4]" "Deployment Completed (Pure Ingress Access Mode)"
echo -e "\033[1;32mDeployment successful via Terraform & NGINX Ingress!\033[0m"
echo -e "Access the Gateway via Ingress:"
echo -e "  Direct Browser: \033[1;36mhttp://localhost/health\033[0m"
echo -e "  Domain Mode:    \033[1;36mhttp://theme-based-rag-workflow.local/health\033[0m"



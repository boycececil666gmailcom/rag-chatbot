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

echo "Applying Kubernetes manifests..."
kubectl apply -f k8s/namespace.yaml || exit 1
kubectl apply -f k8s/secrets.yaml || exit 1
kubectl apply -f k8s/qdrant-statefulset.yaml || exit 1
kubectl apply -f k8s/backend-deployment.yaml || exit 1
kubectl apply -f k8s/gateway-deployment.yaml || exit 1
kubectl apply -f k8s/ingress.yaml || exit 1

echo "Waiting for deployments to roll out..."
kubectl rollout status statefulset/qdrant -n theme-based-rag-workflow --timeout=90s || exit 1
kubectl rollout status deployment/theme-based-rag-backend -n theme-based-rag-workflow --timeout=90s || exit 1
kubectl rollout status deployment/theme-based-rag-gateway -n theme-based-rag-workflow --timeout=90s || exit 1

# Step 4: Port-forward the gateway service
log_step "[4/4]" "Starting API Gateway Port-Forwarding (port 8080)"
echo "Access the chatbot gateway at http://localhost:8080"
echo "Press Ctrl+C to stop port-forwarding."
kubectl port-forward service/theme-based-rag-gateway-service -n theme-based-rag-workflow 8080:80



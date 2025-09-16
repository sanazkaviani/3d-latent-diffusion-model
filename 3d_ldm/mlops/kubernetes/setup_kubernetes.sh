#!/bin/bash
# Setup script for Kubernetes deployment
# Usage: ./setup_kubernetes.sh [apply|delete|status]

set -e

ACTION=${1:-apply}
NAMESPACE="3d-ldm"
KUBECTL_CMD="kubectl"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 3D LDM Kubernetes Deployment Script${NC}"

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}❌ kubectl is not installed. Please install kubectl first.${NC}"
    exit 1
fi

# Check if cluster is accessible
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}❌ Cannot connect to Kubernetes cluster. Please check your kubeconfig.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Connected to Kubernetes cluster${NC}"

# Function to apply manifests
apply_manifests() {
    echo -e "${YELLOW}📦 Applying Kubernetes manifests...${NC}"
    
    # Create namespace if it doesn't exist
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply manifests in order
    echo -e "${BLUE}📁 Applying storage manifests...${NC}"
    kubectl apply -f mlops/kubernetes/storage.yaml
    
    echo -e "${BLUE}🔐 Applying RBAC manifests...${NC}"
    kubectl apply -f mlops/kubernetes/rbac.yaml
    
    echo -e "${BLUE}⚙️ Applying ConfigMaps and Secrets...${NC}"
    kubectl apply -f mlops/kubernetes/config.yaml
    
    echo -e "${BLUE}🚀 Applying deployment manifests...${NC}"
    kubectl apply -f mlops/kubernetes/deployment.yaml
    
    echo -e "${BLUE}🌐 Applying ingress manifests...${NC}"
    kubectl apply -f mlops/kubernetes/ingress.yaml
    
    echo -e "${BLUE}📊 Applying monitoring manifests...${NC}"
    kubectl apply -f mlops/kubernetes/monitoring.yaml
    
    echo -e "${GREEN}✅ All manifests applied successfully!${NC}"
}

# Function to delete manifests
delete_manifests() {
    echo -e "${YELLOW}🗑️ Deleting Kubernetes manifests...${NC}"
    
    # Delete in reverse order
    kubectl delete -f mlops/kubernetes/monitoring.yaml --ignore-not-found=true
    kubectl delete -f mlops/kubernetes/ingress.yaml --ignore-not-found=true
    kubectl delete -f mlops/kubernetes/deployment.yaml --ignore-not-found=true
    kubectl delete -f mlops/kubernetes/config.yaml --ignore-not-found=true
    kubectl delete -f mlops/kubernetes/rbac.yaml --ignore-not-found=true
    kubectl delete -f mlops/kubernetes/storage.yaml --ignore-not-found=true
    
    # Optionally delete namespace (uncomment if needed)
    # kubectl delete namespace $NAMESPACE --ignore-not-found=true
    
    echo -e "${GREEN}✅ All manifests deleted successfully!${NC}"
}

# Function to show status
show_status() {
    echo -e "${BLUE}📊 Kubernetes Deployment Status${NC}"
    echo -e "${YELLOW}===================================${NC}"
    
    echo -e "\n${BLUE}📁 Namespace:${NC}"
    kubectl get namespace $NAMESPACE 2>/dev/null || echo "Namespace not found"
    
    echo -e "\n${BLUE}💾 Persistent Volumes:${NC}"
    kubectl get pv -l app=3d-ldm 2>/dev/null || echo "No PVs found"
    
    echo -e "\n${BLUE}📦 Persistent Volume Claims:${NC}"
    kubectl get pvc -n $NAMESPACE 2>/dev/null || echo "No PVCs found"
    
    echo -e "\n${BLUE}🚀 Deployments:${NC}"
    kubectl get deployments -n $NAMESPACE 2>/dev/null || echo "No deployments found"
    
    echo -e "\n${BLUE}📱 Pods:${NC}"
    kubectl get pods -n $NAMESPACE 2>/dev/null || echo "No pods found"
    
    echo -e "\n${BLUE}🌐 Services:${NC}"
    kubectl get services -n $NAMESPACE 2>/dev/null || echo "No services found"
    
    echo -e "\n${BLUE}🔗 Ingress:${NC}"
    kubectl get ingress -n $NAMESPACE 2>/dev/null || echo "No ingress found"
    
    echo -e "\n${BLUE}📊 HPA (Horizontal Pod Autoscaler):${NC}"
    kubectl get hpa -n $NAMESPACE 2>/dev/null || echo "No HPA found"
    
    echo -e "\n${BLUE}🔐 Service Accounts:${NC}"
    kubectl get serviceaccounts -n $NAMESPACE 2>/dev/null || echo "No service accounts found"
    
    echo -e "\n${BLUE}⚙️ ConfigMaps:${NC}"
    kubectl get configmaps -n $NAMESPACE 2>/dev/null || echo "No configmaps found"
    
    # Show recent events
    echo -e "\n${BLUE}📋 Recent Events:${NC}"
    kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp' | tail -10 2>/dev/null || echo "No events found"
}

# Function to wait for deployment
wait_for_deployment() {
    echo -e "${YELLOW}⏳ Waiting for deployment to be ready...${NC}"
    
    # Wait for deployments to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/3d-ldm-api -n $NAMESPACE
    kubectl wait --for=condition=available --timeout=300s deployment/3d-ldm-worker -n $NAMESPACE
    
    echo -e "${GREEN}✅ Deployment is ready!${NC}"
    
    # Get service endpoints
    echo -e "\n${BLUE}🌐 Service Endpoints:${NC}"
    kubectl get services -n $NAMESPACE -o wide
    
    # Get ingress information
    echo -e "\n${BLUE}🔗 Ingress Information:${NC}"
    kubectl get ingress -n $NAMESPACE
}

# Function to show logs
show_logs() {
    echo -e "${BLUE}📋 Recent Pod Logs${NC}"
    echo -e "${YELLOW}==================${NC}"
    
    # Get API pod logs
    API_POD=$(kubectl get pods -n $NAMESPACE -l app=3d-ldm-api -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [ ! -z "$API_POD" ]; then
        echo -e "\n${BLUE}🔹 API Pod Logs ($API_POD):${NC}"
        kubectl logs $API_POD -n $NAMESPACE --tail=20
    fi
    
    # Get worker pod logs
    WORKER_POD=$(kubectl get pods -n $NAMESPACE -l app=3d-ldm-worker -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [ ! -z "$WORKER_POD" ]; then
        echo -e "\n${BLUE}🔹 Worker Pod Logs ($WORKER_POD):${NC}"
        kubectl logs $WORKER_POD -n $NAMESPACE --tail=20
    fi
}

# Function to run health checks
health_check() {
    echo -e "${BLUE}🏥 Running Health Checks${NC}"
    echo -e "${YELLOW}========================${NC}"
    
    # Check pod health
    echo -e "\n${BLUE}📱 Pod Health:${NC}"
    kubectl get pods -n $NAMESPACE -o wide
    
    # Check service endpoints
    echo -e "\n${BLUE}🌐 Service Endpoints:${NC}"
    kubectl get endpoints -n $NAMESPACE
    
    # Check if API is responding
    API_SERVICE=$(kubectl get service 3d-ldm-api-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
    if [ ! -z "$API_SERVICE" ]; then
        echo -e "\n${BLUE}🔍 API Health Check:${NC}"
        kubectl run health-check --rm -i --restart=Never --image=curlimages/curl -- curl -s http://$API_SERVICE:8000/health || echo "Health check failed"
    fi
}

# Main script logic
case $ACTION in
    "apply")
        apply_manifests
        wait_for_deployment
        show_status
        ;;
    "delete")
        delete_manifests
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs
        ;;
    "health")
        health_check
        ;;
    "full-status")
        show_status
        show_logs
        health_check
        ;;
    *)
        echo -e "${RED}❌ Unknown action: $ACTION${NC}"
        echo -e "${YELLOW}Usage: $0 [apply|delete|status|logs|health|full-status]${NC}"
        echo -e "  apply:       Apply all Kubernetes manifests"
        echo -e "  delete:      Delete all Kubernetes manifests"
        echo -e "  status:      Show deployment status"
        echo -e "  logs:        Show recent pod logs"
        echo -e "  health:      Run health checks"
        echo -e "  full-status: Show status, logs, and health"
        exit 1
        ;;
esac

echo -e "\n${GREEN}🎉 Script completed successfully!${NC}"
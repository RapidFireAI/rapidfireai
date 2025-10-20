#!/bin/bash
# Docker build and publish script for RapidFire AI

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get version from version.py
VERSION=$(python3 -c "from rapidfireai.version import __version__; print(__version__)")
IMAGE_NAME="rapidfireai"
REGISTRY="${DOCKER_REGISTRY:-}" # Set via env var or leave empty for local

echo -e "${GREEN}Building RapidFire AI Docker Image${NC}"
echo "Version: $VERSION"
echo "Image: $IMAGE_NAME"

# Parse command line arguments
ACTION=${1:-build}

case $ACTION in
  build)
    echo -e "${YELLOW}Building Docker image...${NC}"
    docker build \
      -f Dockerfile \
      -t ${IMAGE_NAME}:latest \
      -t ${IMAGE_NAME}:${VERSION} \
      --build-arg VERSION=${VERSION} \
      .
    echo -e "${GREEN}✓ Build complete${NC}"
    echo "Tags: ${IMAGE_NAME}:latest, ${IMAGE_NAME}:${VERSION}"
    ;;

  test)
    echo -e "${YELLOW}Testing Docker image...${NC}"
    
    # Test 1: Image exists
    if ! docker image inspect ${IMAGE_NAME}:latest >/dev/null 2>&1; then
      echo -e "${RED}✗ Image not found. Run './docker-build.sh build' first${NC}"
      exit 1
    fi
    echo -e "${GREEN}✓ Image exists${NC}"
    
    # Test 2: GPU support
    echo "Testing GPU support..."
    docker run --rm --gpus all ${IMAGE_NAME}:latest python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || true
    
    # Test 3: RapidFire CLI
    echo "Testing RapidFire CLI..."
    docker run --rm ${IMAGE_NAME}:latest rapidfireai --version
    echo -e "${GREEN}✓ CLI works${NC}"
    
    # Test 4: Dependencies
    echo "Testing dependencies..."
    docker run --rm ${IMAGE_NAME}:latest python3 -c "import transformers, peft, trl, mlflow, flask; print('All imports successful')"
    echo -e "${GREEN}✓ Dependencies installed${NC}"
    
    echo -e "${GREEN}✓ All tests passed${NC}"
    ;;

  push)
    if [ -z "$REGISTRY" ]; then
      echo -e "${RED}✗ DOCKER_REGISTRY environment variable not set${NC}"
      echo "Example: export DOCKER_REGISTRY=myregistry.com/myorg"
      exit 1
    fi
    
    echo -e "${YELLOW}Tagging for registry: $REGISTRY${NC}"
    docker tag ${IMAGE_NAME}:latest ${REGISTRY}/${IMAGE_NAME}:latest
    docker tag ${IMAGE_NAME}:${VERSION} ${REGISTRY}/${IMAGE_NAME}:${VERSION}
    
    echo -e "${YELLOW}Pushing to registry...${NC}"
    docker push ${REGISTRY}/${IMAGE_NAME}:latest
    docker push ${REGISTRY}/${IMAGE_NAME}:${VERSION}
    
    echo -e "${GREEN}✓ Push complete${NC}"
    echo "Images available at:"
    echo "  ${REGISTRY}/${IMAGE_NAME}:latest"
    echo "  ${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    ;;

  run)
    echo -e "${YELLOW}Starting RapidFire AI container...${NC}"
    
    # Create required directories
    mkdir -p rapidfire_experiments mlruns logs data
    
    # Check if container already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${IMAGE_NAME}$"; then
      echo "Container already exists. Removing..."
      docker rm -f ${IMAGE_NAME}
    fi
    
    docker run -d \
      --name ${IMAGE_NAME} \
      --gpus all \
      -p 3000:3000 \
      -p 5002:5002 \
      -p 8081:8081 \
      -v $(pwd)/rapidfire_experiments:/app/rapidfire_experiments \
      -v $(pwd)/mlruns:/app/mlruns \
      -v $(pwd)/logs:/app/logs \
      -v $(pwd)/data:/app/data \
      -e RF_API_HOST=0.0.0.0 \
      -e RF_FRONTEND_HOST=0.0.0.0 \
      -e CUDA_VISIBLE_DEVICES=0 \
      ${IMAGE_NAME}:latest
    
    echo -e "${GREEN}✓ Container started${NC}"
    echo "View logs: docker logs -f ${IMAGE_NAME}"
    echo "Dashboard: http://localhost:3000"
    echo "MLflow: http://localhost:5002"
    ;;

  stop)
    echo -e "${YELLOW}Stopping RapidFire AI container...${NC}"
    docker stop ${IMAGE_NAME} || true
    docker rm ${IMAGE_NAME} || true
    echo -e "${GREEN}✓ Container stopped${NC}"
    ;;

  clean)
    echo -e "${YELLOW}Cleaning up Docker resources...${NC}"
    docker stop ${IMAGE_NAME} 2>/dev/null || true
    docker rm ${IMAGE_NAME} 2>/dev/null || true
    docker rmi ${IMAGE_NAME}:latest 2>/dev/null || true
    docker rmi ${IMAGE_NAME}:${VERSION} 2>/dev/null || true
    docker system prune -f
    echo -e "${GREEN}✓ Cleanup complete${NC}"
    ;;

  help|*)
    echo "RapidFire AI Docker Build Script"
    echo ""
    echo "Usage: ./docker-build.sh [command]"
    echo ""
    echo "Commands:"
    echo "  build    Build the Docker image (default)"
    echo "  test     Run tests on the built image"
    echo "  push     Push image to registry (requires DOCKER_REGISTRY env var)"
    echo "  run      Start a container from the image"
    echo "  stop     Stop and remove the running container"
    echo "  clean    Remove image and cleanup Docker resources"
    echo "  help     Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./docker-build.sh build"
    echo "  ./docker-build.sh test"
    echo "  DOCKER_REGISTRY=myregistry.com/rapidfire ./docker-build.sh push"
    echo "  ./docker-build.sh run"
    ;;
esac


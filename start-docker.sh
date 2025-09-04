#!/bin/bash

# RapidFire AI Multi-Service Startup Script (Docker Version)
# This script starts MLflow server, API server, and frontend tracking server

set -e  # Exit on any error

# Configuration
MLFLOW_PORT=${MLFLOW_PORT:-5002}
MLFLOW_HOST=${MLFLOW_HOST:-0.0.0.0}
FRONTEND_PORT=${FRONTEND_PORT:-3000}
FRONTEND_HOST=${FRONTEND_HOST:-0.0.0.0}
API_PORT=${API_PORT:-8080}
API_HOST=${API_HOST:-0.0.0.0}

# Directory paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"
DISPATCHER_DIR="$SCRIPT_DIR/rapidfireai/dispatcher"
FRONTEND_DIR="$SCRIPT_DIR/rapidfireai/frontend"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# PID file to track processes
PID_FILE="rapidfire_pids.txt"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Function to setup Python environment (simplified for Docker)
setup_python_env() {
    print_status "Setting up Python environment in Docker..."
    
    # In Docker, we assume everything is already installed
    # Just verify the installation
    cd "$PROJECT_ROOT"
    
    if python -c "import rapidfireai" 2>/dev/null; then
        print_success "rapidfireai package is available"
    else
        print_error "rapidfireai package not found. Reinstalling..."
        pip install -e .
    fi
    
    cd "$SCRIPT_DIR"  # Return to script directory
    return 0
}

# Function to cleanup processes on exit
cleanup() {
    print_warning "Shutting down services..."
    
    # Kill processes by port (more reliable for MLflow)
    for port in $MLFLOW_PORT $FRONTEND_PORT $API_PORT; do
        local pids=$(lsof -ti :$port 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            print_status "Killing processes on port $port"
            echo "$pids" | xargs kill -TERM 2>/dev/null || true
            sleep 2
            # Force kill if still running
            local remaining_pids=$(lsof -ti :$port 2>/dev/null || true)
            if [[ -n "$remaining_pids" ]]; then
                echo "$remaining_pids" | xargs kill -9 2>/dev/null || true
            fi
        fi
    done
    
    # Clean up tracked PIDs
    if [[ -f "$PID_FILE" ]]; then
        while read -r pid service; do
            if kill -0 "$pid" 2>/dev/null; then
                print_status "Stopping $service (PID: $pid)"
                # Kill process group to get child processes too
                kill -TERM -$pid 2>/dev/null || kill -TERM $pid 2>/dev/null || true
                sleep 1
                # Force kill if still running
                if kill -0 "$pid" 2>/dev/null; then
                    kill -9 -$pid 2>/dev/null || kill -9 $pid 2>/dev/null || true
                fi
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi
    
    # Final cleanup - kill any remaining MLflow or gunicorn processes
    pkill -f "mlflow server" 2>/dev/null || true
    pkill -f "gunicorn.*rapidfireai" 2>/dev/null || true
    
    print_success "All services stopped"
    exit 0
}

# Function to check if a port is available
check_port() {
    local port=$1
    local service=$2
    
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_error "Port $port is already in use. Cannot start $service."
        return 1
    fi
    return 0
}

# Function to wait for service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    local max_attempts=${4:-30}  # Allow custom timeout, default 30 seconds
    local attempt=1
    
    print_status "Waiting for $service to be ready on $host:$port (timeout: ${max_attempts}s)..."
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            print_success "$service is ready!"
            return 0
        fi
        sleep 1
        ((attempt++))
    done
    
    print_error "$service failed to start within expected time (${max_attempts}s)"
    return 1
}

# Function to start MLflow server
start_mlflow() {
    print_status "Starting MLflow server..."
    
    if ! check_port $MLFLOW_PORT "MLflow server"; then
        return 1
    fi
    
    # Create mlruns directory if it doesn't exist
    mkdir -p mlruns
    
    # Start MLflow server in background
    mlflow server \
        --host $MLFLOW_HOST \
        --port $MLFLOW_PORT \
        --backend-store-uri sqlite:///mlflow.db \
        --default-artifact-root ./mlruns &
    
    local mlflow_pid=$!
    echo "$mlflow_pid MLflow" >> "$PID_FILE"
    
    # Wait for MLflow to be ready
    if wait_for_service $MLFLOW_HOST $MLFLOW_PORT "MLflow server"; then
        print_success "MLflow server started (PID: $mlflow_pid)"
        print_status "MLflow UI available at: http://$MLFLOW_HOST:$MLFLOW_PORT"
        return 0
    else
        return 1
    fi
}

# Function to start API server
start_api_server() {
    print_status "Starting API server with Gunicorn..."
    
    # Check if dispatcher directory exists
    if [[ ! -d "$DISPATCHER_DIR" ]]; then
        print_error "Dispatcher directory not found at $DISPATCHER_DIR"
        return 1
    fi
    
    # Check if gunicorn config file exists
    if [[ ! -f "$DISPATCHER_DIR/gunicorn.conf.py" ]]; then
        print_error "gunicorn.conf.py not found in dispatcher directory"
        return 1
    fi
    
    # Create database directory
    print_status "Creating database directory..."
    mkdir -p ~/db
    # Ensure proper permissions
    chmod 755 ~/db
    
    # Change to dispatcher directory and start Gunicorn server
    cd "$DISPATCHER_DIR"
    
    # Set PYTHONPATH to include the project root
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    
    # Start Gunicorn server in background
    gunicorn -c gunicorn.conf.py &
    
    local api_pid=$!
    cd "$SCRIPT_DIR"  # Return to original directory
    echo "$api_pid API_Server" >> "$PID_FILE"
    
    # Wait for API server to be ready
    if wait_for_service $API_HOST $API_PORT "API server"; then
        print_success "API server started (PID: $api_pid)"
        print_status "API server available at: http://$API_HOST:$API_PORT"
        return 0
    else
        return 1
    fi
}

# Function to build and start frontend server
start_frontend() {
    print_status "Starting frontend tracking server..."
    
    if ! check_port $FRONTEND_PORT "Frontend server"; then
        return 1
    fi
    
    # Check if frontend directory exists
    if [[ ! -d "$FRONTEND_DIR" ]]; then
        print_error "Frontend directory not found at $FRONTEND_DIR"
        return 1
    fi
    
    # Change to frontend directory
    cd "$FRONTEND_DIR"
    
    # Check if we should use Node.js (preferred) or Docker
    if [[ -f "server.js" ]]; then
        print_status "Starting frontend with Node.js directly..."
        
        # Start Node.js server with npm start in background
        print_status "Starting development server with npm start..."
        
        # Write logs to a location that isn't mounted
        LOG_FILE="/tmp/frontend.log"
        print_status "Frontend logs will be written to: $LOG_FILE"
        PORT=$FRONTEND_PORT nohup npm start > "$LOG_FILE" 2>&1 &
        
        local frontend_pid=$!
        cd "$SCRIPT_DIR"  # Return to original directory
        echo "$frontend_pid Frontend_Node" >> "$PID_FILE"
        
        # Wait for frontend to be ready with longer timeout for development server
        if wait_for_service localhost $FRONTEND_PORT "Frontend server" 120; then
            print_success "Frontend server started with Node.js (PID: $frontend_pid)"
            print_status "Frontend available at: http://localhost:$FRONTEND_PORT"
            return 0
        else
            print_error "Frontend development server failed to start. Showing recent logs:"
            if [[ -f "/tmp/frontend.log" ]]; then
                echo "=== Last 20 lines of frontend.log ==="
                tail -20 "/tmp/frontend.log"
                echo "=== End of logs ==="
            else
                print_error "No frontend.log file found"
            fi
            return 1
        fi
    else
        print_error "server.js not found for frontend"
        cd "$SCRIPT_DIR"
        return 1
    fi
}

# Function to display running services
show_status() {
    print_status "RapidFire AI Services Status:"
    echo "=================================="
    
    if [[ -f "$PID_FILE" ]]; then
        while read -r pid service; do
            if kill -0 "$pid" 2>/dev/null; then
                print_success "$service is running (PID: $pid)"
            else
                print_error "$service is not running (PID: $pid)"
            fi
        done < "$PID_FILE"
    else
        print_warning "No services are currently tracked"
    fi
    
    echo ""
    print_status "Available endpoints:"
    echo "- MLflow UI: http://$MLFLOW_HOST:$MLFLOW_PORT"
    echo "- Frontend: http://$FRONTEND_HOST:$FRONTEND_PORT"
    echo "- API Server: http://$API_HOST:$API_PORT"
}

# Main execution
main() {
    print_status "Starting RapidFire AI services in Docker..."
    
    # Remove old PID file
    rm -f "$PID_FILE"
    
    # Set up signal handlers for cleanup
    trap cleanup SIGINT SIGTERM EXIT
    
    # Check for required commands
    for cmd in mlflow gunicorn; do
        if ! command -v $cmd &> /dev/null; then
            print_error "$cmd is not installed or not in PATH"
            exit 1
        fi
    done
    
    # Setup Python environment
    if ! setup_python_env; then
        print_error "Failed to setup Python environment"
        exit 1
    fi
    
    # Start services
    if start_mlflow && start_api_server && start_frontend; then
        print_success "All services started successfully!"
        show_status
        
        print_status "Press Ctrl+C to stop all services"
        
        # Keep script running and monitor processes
        while true; do
            sleep 5
            # Check if any process died
            if [[ -f "$PID_FILE" ]]; then
                while read -r pid service; do
                    if ! kill -0 "$pid" 2>/dev/null; then
                        print_error "$service (PID: $pid) has stopped unexpectedly"
                    fi
                done < "$PID_FILE"
            fi
        done
    else
        print_error "Failed to start one or more services"
        cleanup
        exit 1
    fi
}

# Handle command line arguments
case "${1:-start}" in
    "start")
        main
        ;;
    "stop")
        cleanup
        ;;
    "status")
        show_status
        ;;
    "restart")
        cleanup
        sleep 2
        main
        ;;
    "setup")
        setup_python_env
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart|setup}"
        echo "  start   - Start all services (default)"
        echo "  stop    - Stop all services"
        echo "  status  - Show service status"
        echo "  restart - Restart all services"
        echo "  setup   - Setup Python environment only"
        exit 1
        ;;
esac 

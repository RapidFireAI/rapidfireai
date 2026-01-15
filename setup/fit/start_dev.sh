#!/bin/bash

# RapidFire AI Multi-Service Startup Script
# This script starts MLflow server, API server, and frontend tracking server
# Used specifically for local development mode

set -e  # Exit on any error

# Configuration
RF_MLFLOW_PORT=${RF_MLFLOW_PORT:=8852}
RF_MLFLOW_HOST=${RF_MLFLOW_HOST:=127.0.0.1}
RF_FRONTEND_PORT=${RF_FRONTEND_PORT:=8853}
RF_FRONTEND_HOST=${RF_FRONTEND_HOST:=0.0.0.0}
# API server configuration - these should match DispatcherConfig in constants.py
RF_API_PORT=${RF_API_PORT:=8851}
RF_API_HOST=${RF_API_HOST:=127.0.0.1}

RF_DB_PATH="${RF_DB_PATH:=$HOME/db}"
# Directory paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# Navigate to rapidfireai/fit directory from setup/fit directory
RAPIDFIRE_FIT_DIR="$PROJECT_ROOT/rapidfireai/fit"
DISPATCHER_DIR="$RAPIDFIRE_FIT_DIR/dispatcher"
FRONTEND_DIR="$RAPIDFIRE_FIT_DIR/frontend"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# PID file to track processes
RF_PID_FILE="${RF_PID_FILE:=$HOME/rapidfire_pids.txt}"

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

# Function to setup Python environment
setup_python_env() {
    print_status "Setting up Python environment..."

    # Check if we're in a virtual environment
    # Check multiple indicators: VIRTUAL_ENV env var, conda environment, python path, and pip path
    local in_venv=false

    if [[ -n "$VIRTUAL_ENV" ]]; then
        in_venv=true
        print_status "Detected virtual environment: $VIRTUAL_ENV"
    elif [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        in_venv=true
        print_status "Detected conda environment: $CONDA_DEFAULT_ENV"
    elif [[ "$(which python 2>/dev/null)" == *"venv"* ]] || [[ "$(which python 2>/dev/null)" == *"virtualenv"* ]]; then
        in_venv=true
        print_status "Detected virtual environment via Python path"
    elif [[ "$(which pip 2>/dev/null)" == *"venv"* ]] || [[ "$(which pip 2>/dev/null)" == *"virtualenv"* ]]; then
        in_venv=true
        print_status "Detected virtual environment via pip path"
    elif [[ "$(which python 2>/dev/null)" == *"conda"* ]] || [[ "$(which python 2>/dev/null)" == *"miniconda"* ]]; then
        in_venv=true
        print_status "Detected conda environment via Python path"
    elif [[ "$(which pip 2>/dev/null)" == *"conda"* ]] || [[ "$(which pip 2>/dev/null)" == *"miniconda"* ]]; then
        in_venv=true
        print_status "Detected conda environment via pip path"
    fi

    if [[ "$in_venv" == "false" ]]; then
        print_warning "Not in a virtual environment. This may cause permission issues."
        print_status "Attempting to install with --user flag to avoid permission issues..."
        PIP_USER_FLAG="--user"
    else
        PIP_USER_FLAG=""
    fi

    # Debug information
    print_status "Environment info:"
    print_status "  VIRTUAL_ENV: ${VIRTUAL_ENV:-'not set'}"
    print_status "  CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-'not set'}"
    print_status "  Python path: $(which python 2>/dev/null || echo 'not found')"
    print_status "  Pip path: $(which pip 2>/dev/null || echo 'not found')"
    print_status "  Using --user flag: $([[ -n "$PIP_USER_FLAG" ]] && echo 'yes' || echo 'no')"

    # Install rapidfireai in development mode
    cd "$PROJECT_ROOT"
    print_status "Installing rapidfireai in development mode..."

    if pip install -e . $PIP_USER_FLAG; then
        print_success "rapidfireai package installed successfully"
    else
        print_error "Failed to install rapidfireai package"
        print_warning "If you're getting permission errors, try:"
        print_warning "1. Activate a virtual environment: python3 -m venv .venv && source .venv/bin/activate"
        print_warning "2. Or run with sudo (not recommended): sudo pip install -e ."
        return 1
    fi

    # Install requirements
    if [[ -f "requirements.txt" ]]; then
        print_status "Installing Python requirements..."
        if pip install -r requirements.txt $PIP_USER_FLAG; then
            print_success "Python requirements installed successfully"
        else
            print_error "Failed to install Python requirements"
            return 1
        fi
    fi

    cd "$SCRIPT_DIR"  # Return to script directory
    return 0
}

# Function to cleanup processes on exit
cleanup() {
    print_warning "Shutting down services..."

    # Stop Docker containers if running
    if docker ps --format "table {{.Names}}" | grep -q "rapidfire-frontend"; then
        print_status "Stopping Docker container: rapidfire-frontend"
        docker stop rapidfire-frontend >/dev/null 2>&1 || true
        docker rm rapidfire-frontend >/dev/null 2>&1 || true
    fi

    # Kill processes by port (more reliable for MLflow)
    for port in $RF_MLFLOW_PORT $RF_FRONTEND_PORT $RF_API_PORT; do
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
    if [[ -f "$RF_PID_FILE" ]]; then
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
        done < "$RF_PID_FILE"
        rm -f "$RF_PID_FILE"
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

    if ! check_port $RF_MLFLOW_PORT "MLflow server"; then
        return 1
    fi

    # Start MLflow server in background with process group
    # Use setsid on Linux, nohup on macOS
    if command -v setsid &> /dev/null; then
        setsid mlflow server \
            --host $RF_MLFLOW_HOST \
            --port $RF_MLFLOW_PORT \
            --backend-store-uri sqlite:///${RF_DB_PATH}/rapidfire_mlflow.db > /dev/null 2>&1 &
    else
        nohup mlflow server \
            --host $RF_MLFLOW_HOST \
            --port $RF_MLFLOW_PORT \
            --backend-store-uri sqlite:///${RF_DB_PATH}/rapidfire_mlflow.db > /dev/null 2>&1 &
    fi

    local mlflow_pid=$!
    echo "$mlflow_pid MLflow" >> "$RF_PID_FILE"

    # Wait for MLflow to be ready
    if wait_for_service $RF_MLFLOW_HOST $RF_MLFLOW_PORT "MLflow server"; then
        print_success "MLflow server started (PID: $mlflow_pid)"
        print_status "MLflow UI available at: http://$RF_MLFLOW_HOST:$RF_MLFLOW_PORT"
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
    echo "$api_pid API_Server" >> "$RF_PID_FILE"

    # Wait for API server to be ready
    if wait_for_service $RF_API_HOST $RF_API_PORT "API server"; then
        print_success "API server started (PID: $api_pid)"
        print_status "API server available at: http://$RF_API_HOST:$RF_API_PORT"
        return 0
    else
        return 1
    fi
}

# Function to build and start frontend server
start_frontend() {
    print_status "Starting frontend tracking server..."

    if ! check_port $RF_FRONTEND_PORT "Frontend server"; then
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
    print_status "Starting frontend with Node.js directly..."

    # Determine which package manager to use
    local yarn_binary=""
    if [[ -f ".yarnrc.yml" ]]; then
        # Check for available yarn releases in the yarn/releases directory
        if [[ -f "yarn/releases/yarn-4.9.1.cjs" ]]; then
            yarn_binary="yarn/releases/yarn-4.9.1.cjs"
        elif [[ -f "yarn/releases/yarn-4.6.0.cjs" ]]; then
            yarn_binary="yarn/releases/yarn-4.6.0.cjs"
        elif [[ -f "yarn/releases/yarn-3.5.0.cjs" ]]; then
            yarn_binary="yarn/releases/yarn-3.5.0.cjs"
        fi
    fi

    # Check if node_modules exists
    if [[ ! -d "node_modules" ]]; then
        print_status "Installing Node.js dependencies..."
        # Check if this is a Yarn 2+ project (has .yarnrc.yml)
        if [[ -f ".yarnrc.yml" ]]; then
            print_status "Using local Yarn binary..."

            if [[ -n "$yarn_binary" ]]; then
                node "$yarn_binary" install || {
                    print_error "Failed to install dependencies with local yarn ($yarn_binary)"
                    cd "$SCRIPT_DIR"
                    return 1
                }
            else
                print_error "No local yarn binary found in yarn/releases/"
                cd "$SCRIPT_DIR"
                return 1
            fi
        elif command -v yarn &> /dev/null; then
            yarn install || {
                print_error "Failed to install dependencies with yarn"
                cd "$SCRIPT_DIR"
                return 1
            }
        else
            npm install || {
                print_error "Failed to install dependencies with npm"
                cd "$SCRIPT_DIR"
                return 1
            }
        fi
    fi

    # Start Node.js server with appropriate package manager
    print_status "Starting development server..."
    print_status "Frontend logs will be written to: $SCRIPT_DIR/frontend.log"

    # Use yarn if available, otherwise fall back to npm
    if [[ -f ".yarnrc.yml" ]] && [[ -n "$yarn_binary" ]]; then
        print_status "Using local Yarn binary to start server..."
        PORT=$RF_FRONTEND_PORT nohup node "$yarn_binary" start > "$SCRIPT_DIR/frontend.log" 2>&1 &
    elif command -v yarn &> /dev/null; then
        print_status "Using system Yarn to start server..."
        PORT=$RF_FRONTEND_PORT nohup yarn start > "$SCRIPT_DIR/frontend.log" 2>&1 &
    else
        print_status "Using npm to start server..."
        PORT=$RF_FRONTEND_PORT nohup npm start > "$SCRIPT_DIR/frontend.log" 2>&1 &
    fi

    local frontend_pid=$!
    cd "$SCRIPT_DIR"  # Return to original directory
    echo "$frontend_pid Frontend_Node" >> "$RF_PID_FILE"

    # Wait for frontend to be ready with longer timeout for development server
    if wait_for_service localhost $RF_FRONTEND_PORT "Frontend server" 120; then
        print_success "Frontend server started with Node.js (PID: $frontend_pid)"
        print_status "Frontend available at: http://localhost:$RF_FRONTEND_PORT"
        return 0
    else
        print_error "Frontend development server failed to start. Showing recent logs:"
        if [[ -f "$SCRIPT_DIR/frontend.log" ]]; then
            echo "=== Last 20 lines of frontend.log ==="
            tail -20 "$SCRIPT_DIR/frontend.log"
            echo "=== End of logs ==="
        else
            print_error "No frontend.log file found"
        fi
        return 1
    fi
}

# Function to display running services
show_status() {
    print_status "RapidFire AI Services Status:"
    echo "=================================="

    if [[ -f "$RF_PID_FILE" ]]; then
        while read -r pid service; do
            if kill -0 "$pid" 2>/dev/null; then
                print_success "$service is running (PID: $pid)"
            else
                print_error "$service is not running (PID: $pid)"
            fi
        done < "$RF_PID_FILE"
    else
        print_warning "No services are currently tracked"
    fi

    # Check Docker container
    if docker ps -q -f name=rapidfire-frontend | grep -q .; then
        print_success "Frontend Docker container is running"
    fi

    echo ""
    print_status "Available endpoints:"
    echo "- MLflow UI: http://$RF_MLFLOW_HOST:$RF_MLFLOW_PORT"
    echo "- Frontend: http://$RF_FRONTEND_HOST:$RF_FRONTEND_PORT"
    echo "- API Server: http://$RF_API_HOST:$RF_API_PORT"
}

# Main execution
main() {
    print_status "Starting RapidFire AI services..."

    # Remove old PID file
    rm -f "$RF_PID_FILE"

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
            if [[ -f "$RF_PID_FILE" ]]; then
                while read -r pid service; do
                    if ! kill -0 "$pid" 2>/dev/null; then
                        print_error "$service (PID: $pid) has stopped unexpectedly"
                    fi
                done < "$RF_PID_FILE"
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

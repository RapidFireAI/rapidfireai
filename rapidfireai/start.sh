#!/bin/bash

# RapidFire AI Multi-Service Startup Script
# This script starts MLflow server, API server, and frontend tracking server
# Used for pip-installed package mode

set -e  # Exit on any error

# Configuration
RF_MLFLOW_PORT=${RF_MLFLOW_PORT:=5002}
RF_MLFLOW_HOST=${RF_MLFLOW_HOST:=127.0.0.1}
RF_FRONTEND_PORT=${RF_FRONTEND_PORT:=3000}
RF_FRONTEND_HOST=${RF_FRONTEND_HOST:=0.0.0.0}
# API server configuration - these should match DispatcherConfig in constants.py
RF_API_PORT=${RF_API_PORT:=8080}
RF_API_HOST=${RF_API_HOST:=127.0.0.1}

RF_DB_PATH="${RF_DB_PATH:=$HOME/db}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# PID file to track processes
RF_PID_FILE="${RF_PID_FILE:=rapidfire_pids.txt}"

# Directory paths for pip-installed package
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISPATCHER_DIR="$SCRIPT_DIR/dispatcher"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

RF_PYTHON_EXECUTABLE=${RF_PYTHON_EXECUTABLE:-python3}
RF_PIP_EXECUTABLE=${RF_PIP_EXECUTABLE:-pip3}

if ! command -v $RF_PYTHON_EXECUTABLE &> /dev/null; then
    RF_PYTHON_EXECUTABLE=python
fi

if ! command -v $RF_PIP_EXECUTABLE &> /dev/null; then
    RF_PIP_EXECUTABLE=pip
fi


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

    # Check if the package can be imported
    print_status "Verifying rapidfireai package availability..."

    if ${RF_PYTHON_EXECUTABLE} -c "import rapidfireai; print('Package imported successfully with ${RF_PYTHON_EXECUTABLE}')" 2>/dev/null; then
        print_success "rapidfireai package is available with ${RF_PYTHON_EXECUTABLE}"
    else
        print_error "rapidfireai package is not available with ${RF_PYTHON_EXECUTABLE}"
        print_warning "Try reinstalling the package: ${RF_PIP_EXECUTABLE} install rapidfireai"
        return 1
    fi

    # Install any missing dependencies
    print_status "Checking for required dependencies..."
    if ${RF_PYTHON_EXECUTABLE} -c "import mlflow, gunicorn, flask" 2>/dev/null; then
        print_success "All required dependencies are available"
    else
        print_warning "Some dependencies may be missing. Installing requirements..."
        ${RF_PIP_EXECUTABLE} install mlflow gunicorn flask flask-cors
    fi

    return 0
}

# Function to cleanup processes on exit
cleanup() {
    print_warning "Shutting down services..."

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

    # Final cleanup - kill any remaining MLflow, gunicorn, or Flask processes
    pkill -f "mlflow server" 2>/dev/null || true
    pkill -f "gunicorn.*rapidfireai" 2>/dev/null || true
    pkill -f "python3.*server.py" 2>/dev/null || true
    pkill -f "python.*server.py" 2>/dev/null || true

    # Additional cleanup for any remaining processes on our ports
    for port in $RF_MLFLOW_PORT $RF_FRONTEND_PORT $RF_API_PORT; do
        local remaining_pids=$(lsof -ti :$port 2>/dev/null || true)
        if [[ -n "$remaining_pids" ]]; then
            print_status "Force killing remaining processes on port $port"
            echo "$remaining_pids" | xargs kill -9 2>/dev/null || true
        fi
    done

    print_success "All services stopped"
    exit 0
}

# Function to check if a port is available
check_port() {
    local port=$1
    local service=$2

    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_error "Port $port is already in use. Cannot start $service."
        print_status "Processes using port $port:"
        lsof -Pi :$port -sTCP:LISTEN 2>/dev/null || true
        return 1
    fi
    return 0
}

# Function to check for common startup issues
check_startup_issues() {
    print_status "Checking for common startup issues..."

    # Check Python version and packages
    if command -v ${RF_PYTHON_EXECUTABLE} &> /dev/null; then
        local python_version=$(${RF_PYTHON_EXECUTABLE} --version 2>&1)
        print_status "Python version: $python_version"

        # Check for required packages
        local missing_packages=()
        for package in mlflow gunicorn flask; do
            if ! ${RF_PYTHON_EXECUTABLE} -c "import $package" 2>/dev/null; then
                missing_packages+=("$package")
            fi
        done

        if [[ ${#missing_packages[@]} -gt 0 ]]; then
            print_warning "Missing packages: ${missing_packages[*]}"
            print_status "Installing missing packages..."
            ${RF_PIP_EXECUTABLE} install "${missing_packages[@]}" || print_error "Failed to install packages"
        fi
    fi

    # Check disk space
    local available_space=$(df . | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 1000000 ]]; then
        print_warning "Low disk space: ${available_space}KB available"
    fi

    # Check if we can write to current directory
    if ! touch "$SCRIPT_DIR/test_write.tmp" 2>/dev/null; then
        print_error "Cannot write to script directory: $SCRIPT_DIR"
        return 1
    fi
    rm -f "$SCRIPT_DIR/test_write.tmp"

    return 0
}

# Function to wait for service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    local max_attempts=${4:-30}  # Allow custom timeout, default 30 seconds
    local attempt=1

    print_status "Waiting for $service to be ready on $host:$port (timeout: ${max_attempts} attempts)..."

    if command -v nc &> /dev/null; then
        ping_command="$(command -v nc) -z $host $port"
    else
        ping_command="$RF_PYTHON_EXECUTABLE -c 'from rapidfireai.utils.ping import ping_server; checker=ping_server(\"${host}\", ${port}); exit(1) if not checker else exit(0)'"
    fi
    while [ $attempt -le $max_attempts ]; do
        if eval ${ping_command} &>/dev/null; then
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
    print_status "Making Database directory $RF_DB_PATH..."
    mkdir -p "$RF_DB_PATH"

    if ! check_port $RF_MLFLOW_PORT "MLflow server"; then
        return 1
    fi

    # Start MLflow server in background with logging
    print_status "MLflow logs will be written to: $SCRIPT_DIR/mlflow.log"

    # Use setsid on Linux, nohup on macOS
    if command -v setsid &> /dev/null; then
        setsid mlflow server \
            --host $RF_MLFLOW_HOST \
            --port $RF_MLFLOW_PORT \
            --backend-store-uri sqlite:///${RF_DB_PATH}/mlflow.db > "$SCRIPT_DIR/mlflow.log" 2>&1 &
    else
        nohup mlflow server \
            --host $RF_MLFLOW_HOST \
            --port $RF_MLFLOW_PORT \
            --backend-store-uri sqlite:///${RF_DB_PATH}/mlflow.db > "$SCRIPT_DIR/mlflow.log" 2>&1 &
    fi

    local mlflow_pid=$!
    echo "$mlflow_pid MLflow" >> "$RF_PID_FILE"

    # Wait for MLflow to be ready
    if wait_for_service $RF_MLFLOW_HOST $RF_MLFLOW_PORT "MLflow server"; then
        print_success "MLflow server started (PID: $mlflow_pid)"
        return 0
    else
        print_error "MLflow server failed to start. Checking for errors..."

        # Check if process is still running
        if ! kill -0 "$mlflow_pid" 2>/dev/null; then
            print_error "MLflow process has died. Checking logs for startup errors:"
        else
            print_error "MLflow process is running but not responding. Checking logs:"
        fi

        if [[ -f "$SCRIPT_DIR/mlflow.log" ]]; then
            echo "=== Last 30 lines of mlflow.log ==="
            tail -30 "$SCRIPT_DIR/mlflow.log"
            echo "=== End of logs ==="
            echo ""

            # Look for specific error patterns
            if grep -q "Error\|Exception\|Traceback\|Failed\|ImportError\|ModuleNotFoundError" "$SCRIPT_DIR/mlflow.log"; then
                print_error "Found error messages in logs:"
                grep -A 5 -B 2 "Error\|Exception\|Traceback\|Failed\|ImportError\|ModuleNotFoundError" "$SCRIPT_DIR/mlflow.log" | head -20
            fi
        else
            print_error "No mlflow.log file found"
        fi

        # Check if there are any Python errors in the process
        if kill -0 "$mlflow_pid" 2>/dev/null; then
            print_status "MLflow process details:"
            ps -p "$mlflow_pid" -o pid,ppid,cmd,etime 2>/dev/null || true
        fi

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

    # Start Gunicorn server in background with logging
    print_status "API server logs will be written to: $SCRIPT_DIR/api.log"
    gunicorn -c gunicorn.conf.py > "$SCRIPT_DIR/api.log" 2>&1 &

    local api_pid=$!
    cd "$SCRIPT_DIR"  # Return to original directory
    echo "$api_pid API_Server" >> "$RF_PID_FILE"

    # Wait for API server to be ready - use longer timeout for API server
    if wait_for_service $RF_API_HOST $RF_API_PORT "API server" 60; then
        print_success "API server started (PID: $api_pid)"
        print_status "API server available at: http://$RF_API_HOST:$RF_API_PORT"
        return 0
    else
        print_error "API server failed to start. Checking for errors..."

        # Check if process is still running
        if ! kill -0 "$api_pid" 2>/dev/null; then
            print_error "API process has died. Checking logs for startup errors:"
        else
            print_error "API process is running but not responding. Checking logs:"
        fi

        if [[ -f "$SCRIPT_DIR/api.log" ]]; then
            echo "=== Last 30 lines of api.log ==="
            tail -30 "$SCRIPT_DIR/api.log"
            echo "=== End of logs ==="
            echo ""

            # Look for specific error patterns
            if grep -q "Error\|Exception\|Traceback\|Failed\|ImportError\|ModuleNotFoundError" "$SCRIPT_DIR/api.log"; then
                print_error "Found error messages in logs:"
                grep -A 5 -B 2 "Error\|Exception\|Traceback\|Failed\|ImportError\|ModuleNotFoundError" "$SCRIPT_DIR/api.log" | head -20
            fi
        else
            print_error "No api.log file found"
        fi

        # Check if there are any Python errors in the process
        if kill -0 "$api_pid" 2>/dev/null; then
            print_status "API process details:"
            ps -p "$api_pid" -o pid,ppid,cmd,etime 2>/dev/null || true
        fi

        return 1
    fi
}

# Function to start frontend server
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

    # Check if build directory exists
    if [[ ! -d "build" ]]; then
        print_error "Build directory not found. Please run 'npm run build' in the frontend directory first."
        cd "$SCRIPT_DIR"
        return 1
    fi

    # Check if Flask server exists
    if [[ ! -f "server.py" ]]; then
        print_error "Flask server (server.py) not found in frontend directory"
        cd "$SCRIPT_DIR"
        return 1
    fi

    # Test if the server can be imported without errors
    print_status "Testing frontend server imports..."
    if ! ${RF_PYTHON_EXECUTABLE} -c "import server" 2>/dev/null; then
        print_error "Frontend server has import errors. Testing with verbose output:"
        ${RF_PYTHON_EXECUTABLE} -c "import server" 2>&1 | head -20
        cd "$SCRIPT_DIR"
        return 1
    fi
    print_success "Frontend server imports successfully"

    print_status "Starting production frontend server with Flask..."

    # Start Flask server in background with process group
    print_status "Frontend logs will be written to: $SCRIPT_DIR/frontend.log"
    cd "$FRONTEND_DIR"

    # Use setsid on Linux, nohup on macOS for better process management
    if command -v setsid &> /dev/null; then
        PORT=$RF_FRONTEND_PORT setsid ${RF_PYTHON_EXECUTABLE} server.py > "$SCRIPT_DIR/frontend.log" 2>&1 &
    else
        PORT=$RF_FRONTEND_PORT nohup ${RF_PYTHON_EXECUTABLE} server.py > "$SCRIPT_DIR/frontend.log" 2>&1 &
    fi

    local frontend_pid=$!
    cd "$SCRIPT_DIR"  # Return to original directory

    # Store both PID and process group ID for better cleanup
    if command -v setsid &> /dev/null; then
        # On Linux, we can get the process group ID
        echo "$frontend_pid Frontend_Flask" >> "$RF_PID_FILE"
    else
        # On macOS, just store the PID
        echo "$frontend_pid Frontend_Flask" >> "$RF_PID_FILE"
    fi

    # Wait for frontend to be ready - check both localhost and 127.0.0.1
    local frontend_ready=false
    local check_hosts=("localhost" "127.0.0.1")

    for host in "${check_hosts[@]}"; do
        if wait_for_service $host $RF_FRONTEND_PORT "Frontend server" 15; then
            print_success "Frontend Flask server started (PID: $frontend_pid) on $host:$RF_FRONTEND_PORT"
            frontend_ready=true
            break
        fi
    done

    if [[ "$frontend_ready" == false ]]; then
        print_error "Frontend Flask server failed to start. Checking for errors..."

        # Check if process is still running
        if ! kill -0 "$frontend_pid" 2>/dev/null; then
            print_error "Frontend process has died. Checking logs for startup errors:"
        else
            print_error "Frontend process is running but not responding. Checking logs:"
        fi

        if [[ -f "$SCRIPT_DIR/frontend.log" ]]; then
            echo "=== Last 30 lines of frontend.log ==="
            tail -30 "$SCRIPT_DIR/frontend.log"
            echo "=== End of logs ==="
            echo ""

            # Look for specific error patterns
            if grep -q "Error\|Exception\|Traceback\|Failed" "$SCRIPT_DIR/frontend.log"; then
                print_error "Found error messages in logs:"
                grep -A 5 -B 2 "Error\|Exception\|Traceback\|Failed" "$SCRIPT_DIR/frontend.log" | head -20
            fi
        else
            print_error "No frontend.log file found"
        fi

        # Check if there are any Python errors in the process
        if kill -0 "$frontend_pid" 2>/dev/null; then
            print_status "Frontend process details:"
            ps -p "$frontend_pid" -o pid,ppid,cmd,etime 2>/dev/null || true
        fi

        return 1
    fi

    return 0
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

    echo ""
    print_success "🚀 RapidFire Frontend is ready!"
    print_status "👉 Open your browser and navigate to: http://$RF_FRONTEND_HOST:$RF_FRONTEND_PORT"
    print_status "   (Click the link above or copy/paste the URL into your browser)"

    # Show log file status
    echo ""
    print_status "Log files:"
    for log_file in "mlflow.log" "api.log" "frontend.log"; do
        if [[ -f "$SCRIPT_DIR/$log_file" ]]; then
            local size=$(du -h "$SCRIPT_DIR/$log_file" | cut -f1)
            print_status "- $log_file: $size"
        else
            print_warning "- $log_file: not found"
        fi
    done
}

# Function to start services based on mode
start_services() {
    local services_started=0
    local total_services=3

    # Start MLflow server
    if start_mlflow; then
        ((services_started++))
    else
        print_error "Failed to start MLflow server"
    fi

    # Start API server
    if start_api_server; then
        ((services_started++))
    else
        print_error "Failed to start API server"
    fi

    # Start frontend server
    if start_frontend; then
        ((services_started++))
    else
        print_error "Failed to start frontend server"
    fi

    return $((total_services - services_started))
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

    # Check for common startup issues
    if ! check_startup_issues; then
        print_error "Startup checks failed"
        exit 1
    fi

    # Start services
    if start_services; then
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

        # Show summary of all log files for debugging
        print_status "=== Startup Failure Summary ==="
        for log_file in "mlflow.log" "api.log" "frontend.log"; do
            if [[ -f "$SCRIPT_DIR/$log_file" ]]; then
                echo ""
                print_status "=== $log_file ==="
                if [[ -s "$SCRIPT_DIR/$log_file" ]]; then
                    tail -10 "$SCRIPT_DIR/$log_file"
                else
                    echo "(empty log file)"
                fi
            fi
        done

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
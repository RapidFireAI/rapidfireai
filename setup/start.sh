#!/bin/bash

# RapidFire AI Multi-Service Startup Script
# This script starts MLflow server, API server, and frontend tracking server
# Used for pip-installed package mode

set -e  # Exit on any error

# Configuration
if [ -z "${COLAB_GPU+x}" ]; then
    RF_HOME="${RF_HOME:=$HOME/rapidfireai}"
else
    RF_HOME="${RF_HOME:=/content/rapidfireai}"
fi
RF_JUPYTER_PORT=${RF_JUPYTER_PORT:=8850}
RF_JUPYTER_HOST=${RF_JUPYTER_HOST:=127.0.0.1}
RF_RAY_PORT=${RF_RAY_PORT:=8855}
RF_RAY_HOST=${RF_RAY_HOST:=127.0.0.1}
RF_MLFLOW_PORT=${RF_MLFLOW_PORT:=8852}
RF_MLFLOW_HOST=${RF_MLFLOW_HOST:=127.0.0.1}
RF_FRONTEND_PORT=${RF_FRONTEND_PORT:=8853}
RF_FRONTEND_HOST=${RF_FRONTEND_HOST:=0.0.0.0}
# API server configuration - these should match DispatcherConfig in constants.py
RF_API_PORT=${RF_API_PORT:=8851}
RF_API_HOST=${RF_API_HOST:=127.0.0.1}

RF_DB_PATH="${RF_DB_PATH:=$RF_HOME/db}"
RF_LOG_PATH="${RF_LOG_PATH:=$RF_HOME/logs}"

RF_TIMEOUT_TIME=${RF_TIMEOUT_TIME:=30}

# Colab mode configuration
if [ -z "${COLAB_GPU+x}" ]; then
    RF_MLFLOW_ENABLED=${RF_MLFLOW_ENABLED:=true}
    RF_TENSORBOARD_ENABLED=${RF_TENSORBOARD_ENABLED:=false}
    RF_TRACKIO_ENABLED=${RF_TRACKIO_ENABLED:=false}
    RF_COLAB_MODE=${RF_COLAB_MODE:=false}
else
    echo "Google Colab environment detected"
    RF_MLFLOW_ENABLED=${RF_MLFLOW_ENABLED:=false}
    RF_TENSORBOARD_ENABLED=${RF_TENSORBOARD_ENABLED:=true}
    RF_TRACKIO_ENABLED=${RF_TRACKIO_ENABLED:=false}
    RF_COLAB_MODE=${RF_COLAB_MODE:=true}
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# PID file to track processes
RF_PID_FILE="${RF_PID_FILE:=$RF_HOME/rapidfire_pids.txt}"

# Directory paths for pip-installed package
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Navigate to rapidfireai/fit directory from setup/fit directory
RAPIDFIRE_DIR="$SCRIPT_DIR/../rapidfireai"
RAPIDFIRE_FIT_DIR="$RAPIDFIRE_DIR/fit"
RAPIDFIRE_EVALS_DIR="$RAPIDFIRE_DIR/evals"
FRONTEND_DIR="$RAPIDFIRE_DIR/frontend"
RAPIDFIRE_MODE=$(cat $RF_HOME/rf_mode.txt 2>/dev/null || echo "fit")
DISPATCHER_DIR="$RAPIDFIRE_DIR/$RAPIDFIRE_MODE/dispatcher"

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
    if [ "$RF_FORCE" != "true" ]; then
        # Confirm cleanup
        read -p "Do you want to shutdown services and delete the PID file? (y/n) " -n 1 -r REPLY
        echo
    else
        REPLY=y
    fi
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Shutting down services..."
    else
        print_warning "Cleanup cancelled"
        exit 0
    fi

    # Kill processes by port (more reliable for MLflow)
    for port in $RF_MLFLOW_PORT $RF_FRONTEND_PORT $RF_API_PORT $RF_JUPYTER_PORT; do
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

    # Final cleanup - ONLY if NOT in Colab mode
    # Colab mode skips this to avoid killing Jupyter/IPython infrastructure
    if [[ "$RF_COLAB_MODE" != "true" ]]; then
        # Safe, specific patterns for non-Colab environments
        pkill -f "mlflow server" 2>/dev/null || true
        pkill -f "gunicorn.*rapidfireai.$RAPIDFIRE_MODE.dispatcher" 2>/dev/null || true
        # Only kill Flask server if we're not in Colab (frontend doesn't run in Colab)
        pkill -f "python.*rapidfireai/frontend/server.py" 2>/dev/null || true
    fi

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

ping_port() {
    local host=$1
    local port=$2
    if command -v nc &> /dev/null; then
        ping_command="$(command -v nc) -z $host $port"
    else
        ping_command="$RF_PYTHON_EXECUTABLE -c 'from rapidfireai.utils.ping import ping_server; checker=ping_server(\"${host}\", ${port}); exit(1) if not checker else exit(0)'"
    fi
    eval $ping_command
    return $?
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

    # Check if we can write to RF_HOME directory
    if ! touch "$RF_HOME/test_write.tmp" 2>/dev/null; then
        print_error "Cannot write to RF_HOME directory: $RF_HOME"
        return 1
    fi
    rm -f "$RF_HOME/test_write.tmp"

    # Check if existing PID file and output contents, error if running
    if [[ -f "$RF_PID_FILE" ]]; then
        print_warning "Existing PID file found: $RF_PID_FILE"
        while read -r pid service; do
            if kill -0 "$pid" 2>/dev/null; then
                print_error "$service is running (PID: $pid)"
                print_error "Try running 'rapidfireai stop' to stop the service"
                return 1
            fi
        done < "$RF_PID_FILE"
    fi

    # Check if ports are available
    if ping_port $RF_MLFLOW_HOST $RF_MLFLOW_PORT; then
        print_error "MLFlow $RF_MLFLOW_HOST:$RF_MLFLOW_PORT in use"
        return 1
    fi
    if ping_port $RF_FRONTEND_HOST $RF_FRONTEND_PORT; then
        print_error "Frontend $RF_FRONTEND_HOST:$RF_FRONTEND_PORT in use"
        return 1
    fi
    if ping_port $RF_API_HOST $RF_API_PORT; then
        print_error "API port $RF_API_HOST:$RF_API_PORT in use"
        return 1
    fi
    return 0
}

# Function to wait for service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    local max_attempts=${4:-$RF_TIMEOUT_TIME}  # Allow custom timeout, default 30 seconds
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
    print_status "MLflow logs will be written to: $RF_LOG_PATH/mlflow.log"

    # Use setsid on Linux, nohup on macOS
    if command -v setsid &> /dev/null; then
        setsid mlflow server \
            --host $RF_MLFLOW_HOST \
            --port $RF_MLFLOW_PORT \
            --backend-store-uri sqlite:///${RF_DB_PATH}/rapidfire_mlflow.db > "$RF_LOG_PATH/mlflow.log" 2>&1 &
    else
        nohup mlflow server \
            --host $RF_MLFLOW_HOST \
            --port $RF_MLFLOW_PORT \
            --backend-store-uri sqlite:///${RF_DB_PATH}/rapidfire_mlflow.db > "$RF_LOG_PATH/mlflow.log" 2>&1 &
    fi

    local mlflow_pid=$!
    echo "$mlflow_pid MLflow" >> "$RF_PID_FILE"

    # Wait for MLflow to be ready
    if wait_for_service $RF_MLFLOW_HOST $RF_MLFLOW_PORT "MLflow server" $RF_TIMEOUT_TIME; then
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

        if [[ -f "$RF_LOG_PATH/mlflow.log" ]]; then
            echo "=== Last 30 lines of mlflow.log ==="
            tail -30 "$RF_LOG_PATH/mlflow.log"
            echo "=== End of logs ==="
            echo ""

            # Look for specific error patterns
            if grep -q "Error\|Exception\|Traceback\|Failed\|ImportError\|ModuleNotFoundError" "$RF_LOG_PATH/mlflow.log"; then
                print_error "Found error messages in logs:"
                grep -A 5 -B 2 "Error\|Exception\|Traceback\|Failed\|ImportError\|ModuleNotFoundError" "$RF_LOG_PATH/mlflow.log" | head -20
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
    # if [[ "$RAPIDFIRE_MODE" != "fit" ]]; then
    #     print_status "Skipping API server (not in fit mode)"
    #     return 0
    # fi

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
    mkdir -p "$RF_DB_PATH"
    # Ensure proper permissions
    chmod 755 "$RF_DB_PATH"

    # Change to dispatcher directory and start Gunicorn server
    cd "$DISPATCHER_DIR"

    # Start Gunicorn server in background with logging
    print_status "API server logs will be written to: $RF_LOG_PATH/api.log"
    gunicorn -c gunicorn.conf.py > "$RF_LOG_PATH/api.log" 2>&1 &

    local api_pid=$!
    cd "$SCRIPT_DIR"  # Return to original directory
    echo "$api_pid API_Server" >> "$RF_PID_FILE"

    # Wait for API server to be ready - use longer timeout for API server
    if wait_for_service $RF_API_HOST $RF_API_PORT "API server" $((RF_TIMEOUT_TIME * 2)); then
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

        if [[ -f "$RF_LOG_PATH/api.log" ]]; then
            echo "=== Last 30 lines of api.log ==="
            tail -30 "$RF_LOG_PATH/api.log"
            echo "=== End of logs ==="
            echo ""

            # Look for specific error patterns
            if grep -q "Error\|Exception\|Traceback\|Failed\|ImportError\|ModuleNotFoundError" "$RF_LOG_PATH/api.log"; then
                print_error "Found error messages in logs:"
                grep -A 5 -B 2 "Error\|Exception\|Traceback\|Failed\|ImportError\|ModuleNotFoundError" "$RF_LOG_PATH/api.log" | head -20
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
    print_status "Frontend logs will be written to: $RF_LOG_PATH/frontend.log"
    cd "$FRONTEND_DIR"

    # Use setsid on Linux, nohup on macOS for better process management
    if command -v setsid &> /dev/null; then
        PORT=$RF_FRONTEND_PORT setsid ${RF_PYTHON_EXECUTABLE} server.py > "$RF_LOG_PATH/frontend.log" 2>&1 &
    else
        PORT=$RF_FRONTEND_PORT nohup ${RF_PYTHON_EXECUTABLE} server.py > "$RF_LOG_PATH/frontend.log" 2>&1 &
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
    local check_hosts=("localhost" "127.0.0.1" "$RF_FRONTEND_HOST")

    for host in "${check_hosts[@]}"; do
        if wait_for_service $host $RF_FRONTEND_PORT "Frontend server" $RF_TIMEOUT_TIME; then
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

        if [[ -f "$RF_LOG_PATH/frontend.log" ]]; then
            echo "=== Last 30 lines of frontend.log ==="
            tail -30 "$RF_LOG_PATH/frontend.log"
            echo "=== End of logs ==="
            echo ""

            # Look for specific error patterns
            if grep -q "Error\|Exception\|Traceback\|Failed" "$RF_LOG_PATH/frontend.log"; then
                print_error "Found error messages in logs:"
                grep -A 5 -B 2 "Error\|Exception\|Traceback\|Failed" "$RF_LOG_PATH/frontend.log" | head -20
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

# Function to conditionally start frontend based on mode
start_frontend_if_needed() {
    # In Colab mode, always skip frontend
    if [[ "$RF_COLAB_MODE" == "true" ]]; then
        print_status "⊗ Skipping frontend (using TensorBoard in Colab mode)"
        return 0
    fi

    # Otherwise start frontend
    start_frontend
    return $?
}

# Function to display running services
show_status() {
    # Get mode from rf_mode.txt in RF_HOME
    mode_file="${RF_HOME}/rf_mode.txt"
    if [[ -f "$mode_file" ]]; then
        rf_mode=$(cat "$mode_file")
    else
        rf_mode="unknown"
    fi
    rf_version=$(rapidfireai --version)
    print_status "${rf_version} Services Status, Mode: ${rf_mode}"
    echo "================================================"

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

    # Display appropriate message based on mode
    if [[ "$RF_COLAB_MODE" == "true" ]]; then
        print_success "🚀 RapidFire running in Colab mode!"
        if [[ "$rf_mode" == "fit" ]]; then
            print_status "📊 Use TensorBoard for metrics visualization:"
            print_status "   In a Colab notebook cell, run:"
            print_status "   %tensorboard --logdir $RF_HOME/rapidfire_experiments/tensorboard_logs/{experiment_name}"
        fi
    else
        # if [[ "$rf_mode" == "fit" ]]; then
            if ping_port $RF_FRONTEND_HOST $RF_FRONTEND_PORT; then
                print_success "🚀 RapidFire Frontend is ready!"
                print_status "👉 Open your browser and navigate to: http://$RF_FRONTEND_HOST:$RF_FRONTEND_PORT"
                print_status "   (Click the link above or copy/paste the URL into your browser)"
            else
                print_error "🚨 RapidFire Frontend is not ready!"
            fi
        # fi
    fi
    if [[ "$RF_MLFLOW_ENABLED" == "true" ]]; then
        if ping_port $RF_MLFLOW_HOST $RF_MLFLOW_PORT; then
            print_success "🚀 MLflow server is ready!"
        else
            print_error "🚨 MLflow server is not ready!"
        fi
    fi
    if ping_port $RF_API_HOST $RF_API_PORT; then
        print_success "🚀 RapidFire API server is ready!"
    else
        print_error "🚨 RapidFire API server is not ready!"
    fi
    # if [[ "$rf_mode" == "evals" ]]; then
    #     if ping_port $RF_RAY_HOST $RF_RAY_PORT; then
    #         print_success "🚀 RapidFire Ray server is ready!"
    #     else
    #         print_error "🚨 RapidFire Ray server is not ready!"
    #     fi
    # fi

    # Show log file status
    echo ""
    print_status "Log files:"

    # Always check api.log
    if [[ -f "$RF_LOG_PATH/api.log" ]]; then
        local size=$(du -h "$RF_LOG_PATH/api.log" | cut -f1)
        print_status "- $RF_LOG_PATH/api.log: $size"
    else
        print_warning "- $RF_LOG_PATH/api.log: not found"
    fi

    # Only check mlflow.log if MLflow is running
    if [[ "$RF_MLFLOW_ENABLED" == "true" ]]; then
        if [[ -f "$RF_LOG_PATH/mlflow.log" ]]; then
            local size=$(du -h "$RF_LOG_PATH/mlflow.log" | cut -f1)
            print_status "- $RF_LOG_PATH/mlflow.log: $size"
        else
            print_warning "- $RF_LOG_PATH/mlflow.log: not found"
        fi
    fi

    # Only check frontend.log if frontend is running
    if [[ "$RF_COLAB_MODE" != "true" ]]; then
        if [[ -f "$RF_LOG_PATH/frontend.log" ]]; then
            local size=$(du -h "$RF_LOG_PATH/frontend.log" | cut -f1)
            print_status "- $RF_LOG_PATH/frontend.log: $size"
        else
            print_warning "- $RF_LOG_PATH/frontend.log: not found"
        fi
    fi
}

# Function to start services based on mode
start_services() {
    local services_started=0
    local total_services=1  # API server always runs

    # Calculate total services based on mode
    # MLflow runs unless tensorboard-only in Colab
    if [[ "$RF_MLFLOW_ENABLED" == "true" ]]; then
        ((total_services++))
    fi

    # Frontend runs unless Colab mode
    if [[ "$RF_COLAB_MODE" != "true" ]]; then
        ((total_services++))
    fi

    if [[ ! -d "$RF_LOG_PATH" ]]; then
        print_status "Creating log directory $RF_LOG_PATH..."
        mkdir -p "$RF_LOG_PATH"
    fi

    print_status "Starting $total_services service(s)..."

    # Start MLflow server (conditionally)
    if [[ "$RF_MLFLOW_ENABLED" == "true" ]]; then
        if start_mlflow; then
            ((services_started++))
        else
            print_error "Failed to start MLflow server"
        fi
    fi

    # Start API server (always)
    if start_api_server; then
        ((services_started++))
    else
        print_error "Failed to start API server"
    fi

    # Start frontend server (conditionally)
    if [[ "$RF_COLAB_MODE" != "true" ]]; then
        if start_frontend; then
            ((services_started++))
        else
            print_error "Failed to start frontend server"
        fi
    else
        print_status "⊗ Skipping frontend (using TensorBoard in Colab mode)"
    fi

    return $((total_services - services_started))
}

# Main execution
main() {
    rf_version=$(rapidfireai --version)
    print_status "Starting ${rf_version} services..."

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

    # Remove old PID file
    rm -f "$RF_PID_FILE"

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
            if [[ -f "$RF_LOG_PATH/$log_file" ]]; then
                echo ""
                print_status "=== $log_file ==="
                if [[ -s "$RF_LOG_PATH/$log_file" ]]; then
                    tail -10 "$RF_LOG_PATH/$log_file"
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
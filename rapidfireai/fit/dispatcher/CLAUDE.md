# CLAUDE.md - Dispatcher

This file provides guidance for working with the dispatcher module of RapidFire AI.

## Overview

The dispatcher is a Flask-based REST API that provides the communication layer between the web UI (frontend) and the RapidFire backend. It exposes endpoints for viewing experiment status, retrieving run information, and triggering Interactive Control Operations (IC Ops).

## Files

### dispatcher.py
**Purpose**: Flask application with REST endpoints for UI communication

**Key Responsibilities**:
- Serves REST API for frontend dashboard
- Provides endpoints for run queries and experiment status
- Handles IC Ops requests (stop, resume, clone-modify, delete)
- Returns logs for debugging
- Manages CORS for local development

**Architecture**:
- Flask app with CORS enabled for localhost:3000 (frontend dev server)
- Stateless request handling (reads from database on each request)
- Returns JSON responses
- Error handling with try/catch and HTTP status codes

**Route Categories**:

**Health Check**:
- `GET /dispatcher/health-check`: Server health status

**UI Data Routes**:
- `GET /dispatcher/get-all-runs`: Retrieve all runs for current experiment
- `POST /dispatcher/get-run`: Get single run by ID
- `GET /dispatcher/get-all-experiment-names`: List all experiment names
- `GET /dispatcher/get-running-experiment`: Get currently active experiment

**Interactive Control Routes**:
- `POST /dispatcher/clone-modify-run`: Clone run with optional modifications
- `POST /dispatcher/stop-run`: Stop active run
- `POST /dispatcher/resume-run`: Resume stopped run
- `POST /dispatcher/delete-run`: Delete run (mark as KILLED)

**Log Routes**:
- `POST /dispatcher/get-ic-logs`: Get IC Ops logs
- `POST /dispatcher/get-experiment-logs`: Get experiment logs

**Key Methods**:

`get_all_runs()`:
- Returns list of all runs with status, metrics, config
- Used by dashboard to display run table
- Includes calculated fields (progress, current_chunk, etc.)

`clone_modify_run(run_id, config_leaf, warm_start)`:
- Creates IC Ops request in database for clone
- Controller polls and processes request
- Returns new run_id or error

`stop_run(run_id)`:
- Validates run is in stoppable state (ONGOING)
- Creates IC Ops request in database
- Controller processes asynchronously
- Returns success/error status

`resume_run(run_id)`:
- Validates run is STOPPED
- Creates IC Ops request in database
- Controller adds run back to scheduler
- Returns success/error status

`delete_run(run_id)`:
- Marks run as KILLED
- Creates IC Ops request in database
- Controller removes from scheduler
- Returns success/error status

**Error Handling**:
```python
try:
    # ... operation ...
    return jsonify(result), 200
except Exception as e:
    logger.error(f"Error: {e}")
    return jsonify({"error": str(e)}), 500
```

**CORS Configuration**:
- Allows origins: localhost:3000, localhost
- Required for frontend dev server (separate port from backend)
- Production: frontend built and served from same origin

### gunicorn.conf.py
**Purpose**: Gunicorn server configuration for production deployment

**Key Settings**:
- `workers`: Number of worker processes (default: 4)
- `bind`: Host and port (default: 0.0.0.0:8081)
- `timeout`: Request timeout (default: 120s)
- `loglevel`: Log verbosity (default: info)

**Usage**:
```bash
gunicorn -c rapidfireai/fit/dispatcher/gunicorn.conf.py rapidfireai.fit.dispatcher.dispatcher:app
```

**Production Notes**:
- Multiple workers for load balancing
- Timeout prevents hanging requests
- Access logs for monitoring

## API Endpoints Reference

### GET /dispatcher/health-check
**Response**:
```json
"Dispatcher is up and running"
```

### GET /dispatcher/get-all-runs
**Response**:
```json
[
  {
    "run_id": 1,
    "run_name": "run_1",
    "status": "ONGOING",
    "current_chunk": 5,
    "current_epoch": 0,
    "metrics": "{\"loss\": 0.5, \"accuracy\": 0.9}",
    "config_leaf": {...},
    "source": "USER",
    "parent_run_id": null,
    "warm_start": false
  },
  ...
]
```

### POST /dispatcher/clone-modify-run
**Request**:
```json
{
  "run_id": 1,
  "config_leaf": {"learning_rate": 1e-4},
  "warm_start": true
}
```
**Response**:
```json
{
  "message": "Clone-modify request created",
  "new_run_id": 5
}
```

### POST /dispatcher/stop-run
**Request**:
```json
{
  "run_id": 1
}
```
**Response**:
```json
{
  "message": "Stop request created successfully"
}
```

### POST /dispatcher/resume-run
**Request**:
```json
{
  "run_id": 1
}
```
**Response**:
```json
{
  "message": "Resume request created successfully"
}
```

### POST /dispatcher/delete-run
**Request**:
```json
{
  "run_id": 1
}
```
**Response**:
```json
{
  "message": "Delete request created successfully"
}
```

## Integration with Frontend

Frontend makes HTTP requests to dispatcher:
```typescript
// Example: Get all runs
const response = await fetch('http://localhost:8081/dispatcher/get-all-runs');
const runs = await response.json();

// Example: Stop run
await fetch('http://localhost:8081/dispatcher/stop-run', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({run_id: 5})
});
```

Frontend polls `get-all-runs` periodically to update dashboard.

## Integration with Controller

Dispatcher writes IC Ops requests to database:
```python
# Dispatcher
db.request_stop(run_id)

# Controller (polling loop)
ic_ops = db.get_ic_ops_request()
for op in ic_ops:
    if op['operation'] == 'STOP':
        self._handle_stop(op['run_id'])
        db.mark_ic_ops_completed(op['ic_id'])
```

Asynchronous communication via database (no direct RPC).

## Running Dispatcher

**Development**:
```bash
# Via start_dev.sh (starts all services)
./rapidfireai/fit/start_dev.sh start

# Or manually
python -m flask --app rapidfireai.fit.dispatcher.dispatcher:app run --port 8081
```

**Production** (via start.sh):
```bash
gunicorn -c rapidfireai/fit/dispatcher/gunicorn.conf.py rapidfireai.fit.dispatcher.dispatcher:app
```

**Testing**:
```bash
# Health check
curl http://localhost:8081/dispatcher/health-check

# Get all runs
curl http://localhost:8081/dispatcher/get-all-runs

# Stop run
curl -X POST http://localhost:8081/dispatcher/stop-run \
  -H "Content-Type: application/json" \
  -d '{"run_id": 1}'
```

## Common Patterns

### Adding New Endpoint

1. **Add route in `register_routes()`**:
```python
self.app.add_url_rule(
    f"{route_prefix}/my-endpoint",
    "my_endpoint",
    self.my_endpoint,
    methods=["POST"]
)
```

2. **Implement handler method**:
```python
def my_endpoint(self) -> tuple[Response, int]:
    try:
        data = request.json
        result = self.db.some_operation(data)
        return jsonify(result), 200
    except Exception as e:
        self._get_logger().error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
```

3. **Update frontend to call endpoint**:
```typescript
await fetch('http://localhost:8081/dispatcher/my-endpoint', {
  method: 'POST',
  body: JSON.stringify(data)
});
```

### Debugging Dispatcher Issues

**Check logs**:
```bash
# Dispatcher logs
cat {experiment_path}/logs/dispatcher.log

# Or watch in real-time
tail -f {experiment_path}/logs/dispatcher.log
```

**Test endpoint directly**:
```bash
curl -X POST http://localhost:8081/dispatcher/stop-run \
  -H "Content-Type: application/json" \
  -d '{"run_id": 1}' \
  -v
```

**Check database state**:
```bash
sqlite3 rapidfire.db "SELECT * FROM interactive_control ORDER BY created_at DESC LIMIT 5;"
```

**CORS issues**:
- Ensure frontend origin in `CORS_ALLOWED_ORIGINS`
- Check browser console for CORS errors
- Verify preflight OPTIONS requests succeed

### Error Handling Best Practices

```python
def endpoint(self) -> tuple[Response, int]:
    try:
        # Validate input
        data = request.json
        if not data or 'run_id' not in data:
            return jsonify({"error": "run_id required"}), 400

        # Perform operation
        result = self.db.some_operation(data['run_id'])

        # Check result
        if not result:
            return jsonify({"error": "Operation failed"}), 500

        return jsonify(result), 200
    except DBException as e:
        self._get_logger().error(f"DB error: {e}")
        return jsonify({"error": "Database error"}), 500
    except Exception as e:
        self._get_logger().error(f"Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500
```

## Performance Considerations

- Dispatcher is stateless (scales horizontally with Gunicorn workers)
- Database is bottleneck for high request volume (SQLite)
- Frontend should throttle polling (e.g., 1-2 second intervals)
- Large run counts may slow `get-all-runs` (consider pagination)

## Security Notes

- No authentication (assumes local/trusted network)
- No rate limiting (relies on frontend behavior)
- CORS restricted to localhost (production should tighten)
- Input validation minimal (assumes trusted clients)

For production deployment, consider adding:
- API authentication (tokens, JWT)
- Rate limiting (Flask-Limiter)
- Input validation (marshmallow, pydantic)
- HTTPS (reverse proxy like nginx)

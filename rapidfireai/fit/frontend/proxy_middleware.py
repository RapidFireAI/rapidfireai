"""
Proxy middleware for RapidFire AI frontend server.
Replicates the functionality of setupProxy.js in Python.
"""

import os
import json
import logging
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse, urljoin
import requests
from flask import Flask, request, Response, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
from rapidfireai.utils.constants import DispatcherConfig, MLFlowConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserProxyManager:
    """Manages user-specific proxy configurations."""
    
    def __init__(self):
        self.user_proxies = {}
        self.default_proxy = {
            'main_proxy_target': MLFlowConfig.URL,
            'static_proxy_target': MLFlowConfig.URL,
            'dispatcher_proxy_target': DispatcherConfig.URL,
        }
    
    def get_user_proxy(self, user_id: str) -> Dict[str, str]:
        """Get or create user-specific proxy configuration."""
        if user_id not in self.user_proxies:
            self.user_proxies[user_id] = self.default_proxy.copy()
        return self.user_proxies[user_id]
    
    def update_proxy_target(self, user_id: str, target_type: str, new_target: str) -> Dict[str, any]:
        """Update a proxy target for a specific user."""
        if not self._is_valid_url(new_target):
            return {
                'success': False,
                'message': 'Invalid URL format',
                'target': self.get_user_proxy(user_id).get(target_type, '')
            }
        
        try:
            url = urlparse(new_target)
            if url.scheme and url.netloc:
                user_proxy = self.get_user_proxy(user_id)
                user_proxy[target_type] = new_target
                return {
                    'success': True,
                    'message': f'{target_type} proxy target updated to {new_target}',
                    'target': new_target
                }
        except Exception as e:
            logger.error(f"Error parsing URL: {e}")
            return {
                'success': False,
                'message': 'Invalid URL format',
                'target': self.get_user_proxy(user_id).get(target_type, '')
            }
        
        return {
            'success': False,
            'message': 'Invalid or missing target URL',
            'target': self.get_user_proxy(user_id).get(target_type, '')
        }
    
    def _is_valid_url(self, url_string: str) -> bool:
        """Check if a string is a valid URL."""
        try:
            result = urlparse(url_string)
            return all([result.scheme, result.netloc])
        except Exception:
            return False


class ProxyMiddleware:
    """Flask middleware for handling proxy requests."""
    
    def __init__(self, app: Flask):
        self.app = app
        self.proxy_manager = UserProxyManager()
        self.setup_routes()
    
    def setup_routes(self):
        """Setup proxy configuration endpoints."""
        
        @self.app.route('/set-main-proxy')
        def set_main_proxy():
            """Set the main proxy target."""
            target = request.args.get('target')
            user_id = request.headers.get('x-user-id', 'default')
            
            result = self.proxy_manager.update_proxy_target(
                user_id, 'main_proxy_target', target
            )
            
            status_code = 200 if result['success'] else 400
            return jsonify(result), status_code
        
        @self.app.route('/set-static-proxy')
        def set_static_proxy():
            """Set the static proxy target."""
            target = request.args.get('target')
            user_id = request.headers.get('x-user-id', 'default')
            
            result = self.proxy_manager.update_proxy_target(
                user_id, 'static_proxy_target', target
            )
            
            status_code = 200 if result['success'] else 400
            return jsonify(result), status_code
        
        @self.app.route('/set-dispatcher-proxy')
        def set_dispatcher_proxy():
            """Set the dispatcher proxy target."""
            target = request.args.get('target')
            user_id = request.headers.get('x-user-id', 'default')
            
            result = self.proxy_manager.update_proxy_target(
                user_id, 'dispatcher_proxy_target', target
            )
            
            status_code = 200 if result['success'] else 400
            return jsonify(result), status_code
        
        @self.app.route('/get-proxy-targets')
        def get_proxy_targets():
            """Get current proxy targets for the user."""
            user_id = request.headers.get('x-user-id', 'default')
            return jsonify(self.proxy_manager.get_user_proxy(user_id))
    
    def should_proxy(self, path: str) -> bool:
        """Determine if a request should be proxied."""
        proxy_paths = [
            '/ajax-api',
            '/get-artifact',
            '/model-versions/get-artifact',
            '/dispatcher'
        ]
        
        return any(path.startswith(proxy_path) for proxy_path in proxy_paths)
    
    def get_proxy_target(self, path: str, user_id: str) -> Optional[str]:
        """Get the appropriate proxy target for a request."""
        user_proxy = self.proxy_manager.get_user_proxy(user_id)
        
        if path.startswith('/ajax-api'):
            return user_proxy['main_proxy_target']
        elif path.startswith('/get-artifact') or path.startswith('/model-versions/get-artifact'):
            return user_proxy['static_proxy_target']
        elif path.startswith('/dispatcher'):
            return user_proxy['dispatcher_proxy_target']
        
        return None
    
    def proxy_request(self, path: str, user_id: str) -> Response:
        """Proxy a request to the appropriate backend service."""
        target = self.get_proxy_target(path, user_id)
        if not target:
            return Response('Proxy target not found', status=404)
        
        try:
            # Prepare the request
            target_url = urljoin(target.rstrip('/') + '/', path.lstrip('/'))
            
            # Get request data
            method = request.method
            headers = dict(request.headers)
            headers.pop('Host', None)  # Remove Host header to avoid conflicts
            
            # Handle query parameters
            if request.query_string:
                target_url += '?' + request.query_string.decode('utf-8')
            
            # Prepare request data
            data = None
            if method in ['POST', 'PUT', 'PATCH']:
                if request.is_json:
                    data = json.dumps(request.get_json())
                    headers['Content-Type'] = 'application/json'
                else:
                    data = request.get_data()
            
            # Make the proxy request
            response = requests.request(
                method=method,
                url=target_url,
                headers=headers,
                data=data,
                stream=True,
                timeout=30
            )
            
            # Create response
            proxy_response = Response(
                response.iter_content(chunk_size=8192),
                status=response.status_code,
                headers=dict(response.headers)
            )
            
            return proxy_response
            
        except requests.exceptions.RequestException as e:
            logger.error(f'Proxy Error: {e}')
            return Response('Proxy Error', status=500)
        except Exception as e:
            logger.error(f'Unexpected error in proxy: {e}')
            return Response('Internal Server Error', status=500)


def setup_proxy(app: Flask):
    """Setup proxy middleware for the Flask app."""
    # Add proxy fix middleware for proper header handling
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
    
    # Create and setup proxy middleware
    proxy_middleware = ProxyMiddleware(app)
    
    # Add before_request handler to intercept proxy requests
    @app.before_request
    def handle_proxy():
        """Handle proxy requests before they reach the main routes."""
        path = request.path
        user_id = request.headers.get('x-user-id', 'default')
        
        if proxy_middleware.should_proxy(path):
            return proxy_middleware.proxy_request(path, user_id)
        
        # Continue with normal request handling
        return None 
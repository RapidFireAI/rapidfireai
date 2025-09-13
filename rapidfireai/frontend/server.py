from flask import Flask, send_from_directory
import os
from pathlib import Path
from proxy_middleware import setup_proxy

# Get the directory where this server.py file is located
current_dir = Path(__file__).parent
build_dir = current_dir / 'build'

app = Flask(__name__, static_folder=str(build_dir))

# Setup proxy middleware
setup_proxy(app)

# Serve React App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(port=3000)
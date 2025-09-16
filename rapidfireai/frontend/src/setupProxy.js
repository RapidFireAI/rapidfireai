// THIS FILE JUST USED FOR LOCAL DEVELOPMENT

const { createProxyMiddleware } = require('http-proxy-middleware');

// Create a simple logger if you don't have a logger utility
const logger = {
    error: (...args) => console.error(...args),
    info: (...args) => console.info(...args),
    warn: (...args) => console.warn(...args),
    debug: (...args) => console.debug(...args)
};

// User-specific proxy mapping
const userProxyMap = new Map();

// Function to get or create user-specific proxy
function getUserProxy(userIdParam) {
  const userId = String(userIdParam); // Create new variable instead of reassigning parameter
  if (!userProxyMap.has(userId)) {
    // Use require('process') or import process from 'process'
    const processEnv = require('process').env;
    userProxyMap.set(userId, {
      mainProxyTarget: 'http://127.0.0.1:5002/',
      staticProxyTarget: 'http://127.0.0.1:5002/',
      dispatcherProxyTarget: 'http://127.0.0.1:8080/',
    });
  }
  return userProxyMap.get(userId);
}

function isValidUrl(string) {
  try {
    new URL(string);
    return true;
  } catch (_) {
    return false;
  }
}

function createUserConfigurableProxy(targetType, wsSupport = false) {
  return createProxyMiddleware({
    target: 'http://placeholder', // Will be overwritten in router
    changeOrigin: true,
    ws: wsSupport,
    router: (req) => {
      const userId = req.headers['x-user-id'] || 'default';
      const userProxy = getUserProxy(userId);
      return userProxy[targetType];
    },
    onError: (err, req, res) => {
      logger.error('Proxy Error:', err);
      res.status(500).send('Proxy Error');
    },
  });
}

function updateProxyTarget(newTarget, originalTarget, targetName) {
  // Create new variable instead of reassigning parameter
  let updatedTarget = originalTarget;
  
  if (newTarget && isValidUrl(newTarget)) {
    try {
      const url = new URL(newTarget);
      if (url.protocol && url.host) {
        updatedTarget = newTarget;
        return { success: true, message: `${targetName} proxy target updated to ${newTarget}`, target: updatedTarget };
      }
    } catch (error) {
      logger.error(`Error parsing URL: ${error.message}`);
      return { success: false, message: 'Invalid URL format', target: updatedTarget };
    }
  }
  return { success: false, message: 'Invalid or missing target URL', target: updatedTarget };
}

// Export directly as an anonymous function (matching the working version)
module.exports = function(app) {
    // Middleware to extract user ID from request
    app.use((req, res, next) => {
        req.userId = String(req.headers['x-user-id'] || 'default');
        next();
    });

    // Main API proxy
    app.use('/ajax-api', createUserConfigurableProxy('mainProxyTarget'));

    // Static content proxies
    app.use('/get-artifact', createUserConfigurableProxy('staticProxyTarget', false));
    app.use('/model-versions/get-artifact', createUserConfigurableProxy('staticProxyTarget', false));

    // Dispatcher proxy
    app.use('/dispatcher', createUserConfigurableProxy('dispatcherProxyTarget'));

    // Endpoint to change the main proxy target at runtime
    app.get('/set-main-proxy', (req, res) => {
        const { target } = req.query;
        const userProxy = getUserProxy(req.userId);
        const result = updateProxyTarget(target, userProxy.mainProxyTarget, 'Main');
        userProxy.mainProxyTarget = result.target;
        res.status(result.success ? 200 : 400).send(result.message);
    });

    // Endpoint to change the static proxy target at runtime
    app.get('/set-static-proxy', (req, res) => {
        const { target } = req.query;
        const userProxy = getUserProxy(req.userId);
        const result = updateProxyTarget(target, userProxy.staticProxyTarget, 'Static');
        userProxy.staticProxyTarget = result.target;
        res.status(result.success ? 200 : 400).send(result.message);
    });

    // Endpoint to change the dispatcher proxy target at runtime
    app.get('/set-dispatcher-proxy', (req, res) => {
        const { target } = req.query;
        const userProxy = getUserProxy(req.userId);
        const result = updateProxyTarget(target, userProxy.dispatcherProxyTarget, 'Dispatcher');
        userProxy.dispatcherProxyTarget = result.target;
        res.status(result.success ? 200 : 400).send(result.message);
    });

    // Endpoint to get the current proxy targets
    app.get('/get-proxy-targets', (req, res) => {
        const userProxy = getUserProxy(req.userId);
        res.json(userProxy);
    });
};

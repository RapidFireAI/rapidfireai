const { createProxyMiddleware } = require('http-proxy-middleware');
// Get RF_API_HOST and RF_API_PORT from environment variables
const rfApiHost: string = process.env.RF_API_HOST || 'localhost';
const rfApiPort: string = process.env.RF_API_PORT || '8851';
const rfApiUrl: string = `http://${rfApiHost}:${rfApiPort}/`;
  // Get RF_MLFLOW_HOST and RF_MLFLOW_PORT from environment variables
const rfMlflowHost: string = process.env.RF_MLFLOW_HOST || 'localhost';
const rfMlflowPort: string = process.env.RF_MLFLOW_PORT || '8852';
const rfMlflowUrl: string = `http://${rfMlflowHost}:${rfMlflowPort}/`;
// eslint-disable-next-line
module.exports = function (app) {
  // The MLflow Gunicorn server is running on port 8852, so we should redirect server requests
  // (eg /ajax-api) to that port.
  // Exception: If the caller has specified an MLFLOW_PROXY, we instead forward server requests
  // there.
  // eslint-disable-next-line no-undef
  // const proxyTarget = process.env.MLFLOW_PROXY || 'http://localhost:8852/';
  const proxyTarget = rfMlflowUrl;
  // eslint-disable-next-line no-undef
  const proxyStaticTarget = process.env.MLFLOW_STATIC_PROXY || proxyTarget;
  app.use(
    createProxyMiddleware('/ajax-api', {
      target: proxyTarget,
      changeOrigin: true,
    }),
  );
  app.use(
    createProxyMiddleware('/graphql', {
      target: proxyTarget,
      changeOrigin: true,
    }),
  );
  app.use(
    createProxyMiddleware('/get-artifact', {
      target: proxyStaticTarget,
      ws: true,
      changeOrigin: true,
    }),
  );
  app.use(
    createProxyMiddleware('/model-versions/get-artifact', {
      target: proxyStaticTarget,
      ws: true,
      changeOrigin: true,
    }),
  );
  app.use(
    createProxyMiddleware('/dispatcher', {
      target: rfApiUrl,
      changeOrigin: true,
    }),
  );
};

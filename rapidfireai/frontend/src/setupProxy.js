const { createProxyMiddleware } = require('http-proxy-middleware');

// eslint-disable-next-line
module.exports = function (app) {
  // The MLflow Gunicorn server is running on port 5000, so we should redirect server requests
  // (eg /ajax-api) to that port.
  // Exception: If the caller has specified an MLFLOW_PROXY, we instead forward server requests
  // there.
  // eslint-disable-next-line no-undef
  // const proxyTarget = process.env.MLFLOW_PROXY || 'http://localhost:5000/';
  let mlflowHost
  if (process.env.RF_MLFLOW_HOST === '0.0.0.0' || !process.env.RF_MLFLOW_HOST) {
    mlflowHost = 'localhost';
  } else {
    mlflowHost = process.env.RF_MLFLOW_HOST;
  }
  let apiHost
  if (process.env.RF_API_HOST === '0.0.0.0' || !process.env.RF_API_HOST) {
    apiHost = 'localhost';
  } else {
    apiHost = process.env.RF_API_HOST;
  }
  const proxyTarget = `http://${mlflowHost}:${parseInt(process.env.RF_MLFLOW_PORT,10)||5002}/`;
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
      target: `http://${apiHost}:${parseInt(process.env.RF_API_PORT, 10)||8080}/`,
      changeOrigin: true,
    }),
  );
};

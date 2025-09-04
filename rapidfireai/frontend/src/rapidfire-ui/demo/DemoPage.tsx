import React from 'react';
import RightSlidingDrawerExample from '../examples/RightSlidingDrawerExample';

const DemoPage: React.FC = () => {
  return (
    <div style={{ 
      minHeight: '100vh', 
      backgroundColor: '#f5f5f5',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      <div style={{ 
        backgroundColor: 'white', 
        borderBottom: '1px solid #e8e8e8',
        padding: '16px 24px',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)'
      }}>
        <h1 style={{ margin: 0, color: '#1890ff', fontSize: '24px' }}>
          RapidFire UI Components Demo
        </h1>
        <p style={{ margin: '8px 0 0 0', color: '#666' }}>
          A showcase of custom UI components for RapidFire applications
        </p>
      </div>
      
      <div style={{ padding: '24px' }}>
        <RightSlidingDrawerExample />
      </div>
    </div>
  );
};

export default DemoPage; 
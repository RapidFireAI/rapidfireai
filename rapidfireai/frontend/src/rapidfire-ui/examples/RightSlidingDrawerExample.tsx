import React, { useState } from 'react';
import { Button, Typography } from '@databricks/design-system';
import RightSlidingDrawer from '../components/RightSlidingDrawer';

const RightSlidingDrawerExample: React.FC = () => {
  const [isBasicOpen, setIsBasicOpen] = useState(false);
  const [isWideOpen, setIsWideOpen] = useState(false);
  const [isCustomOpen, setIsCustomOpen] = useState(false);
  const [isCustomHeaderOpen, setIsCustomHeaderOpen] = useState(false);

  const cardStyle = {
    border: '1px solid #e8e8e8',
    borderRadius: '6px',
    padding: '16px',
    backgroundColor: 'white',
    boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
    marginBottom: '16px'
  };

  const cardTitleStyle = {
    fontSize: '16px',
    fontWeight: '600',
    marginBottom: '12px',
    color: '#333',
    borderBottom: '1px solid #f0f0f0',
    paddingBottom: '8px'
  };

  return (
    <div style={{ padding: '20px', maxWidth: '800px' }}>
      <Typography.Title level={2}>RightSlidingDrawer Examples</Typography.Title>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '24px', width: '100%' }}>
        {/* Basic Example */}
        <div style={cardStyle}>
          <div style={cardTitleStyle}>Basic Drawer</div>
          <Typography.Paragraph>
            A simple drawer with default settings (400px width, backdrop, ESC to close).
          </Typography.Paragraph>
          <Button 
            type="primary" 
            onClick={() => setIsBasicOpen(true)}
            componentId="basic-drawer-button"
          >
            Open Basic Drawer
          </Button>
          
          <RightSlidingDrawer
            isOpen={isBasicOpen}
            onClose={() => setIsBasicOpen(false)}
          >
            <Typography.Title level={3}>Basic Drawer</Typography.Title>
            <Typography.Paragraph>
              This is a basic drawer with default settings. It includes:
            </Typography.Paragraph>
            <ul>
              <li>Default width (400px)</li>
              <li>Backdrop with click-to-close</li>
              <li>ESC key to close</li>
              <li>Body scroll disabled when open</li>
            </ul>
            <hr style={{ margin: '20px 0', border: 'none', borderTop: '1px solid #e8e8e8' }} />
            <Typography.Paragraph>
              You can put any content here - forms, lists, settings, etc.
            </Typography.Paragraph>
          </RightSlidingDrawer>
        </div>

        {/* Wide Drawer Example */}
        <div style={cardStyle}>
          <div style={cardTitleStyle}>Wide Drawer</div>
          <Typography.Paragraph>
            A wider drawer (600px) for content that needs more space.
          </Typography.Paragraph>
          <Button 
            type="primary" 
            onClick={() => setIsWideOpen(true)}
            componentId="wide-drawer-button"
          >
            Open Wide Drawer
          </Button>
          
          <RightSlidingDrawer
            isOpen={isWideOpen}
            onClose={() => setIsWideOpen(false)}
            width={600}
          >
            <Typography.Title level={3}>Wide Drawer</Typography.Title>
            <Typography.Paragraph>
              This drawer is 600px wide, providing more space for complex content.
            </Typography.Paragraph>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
              <div style={{ ...cardStyle, marginBottom: '0' }}>
                <div style={cardTitleStyle}>Left Column</div>
                <Typography.Paragraph>
                  You can create multi-column layouts or side-by-side content.
                </Typography.Paragraph>
              </div>
              <div style={{ ...cardStyle, marginBottom: '0' }}>
                <div style={cardTitleStyle}>Right Column</div>
                <Typography.Paragraph>
                  Perfect for forms, settings panels, or detailed views.
                </Typography.Paragraph>
              </div>
            </div>
          </RightSlidingDrawer>
        </div>

        {/* Custom Styled Drawer */}
        <div style={cardStyle}>
          <div style={cardTitleStyle}>Custom Styled Drawer</div>
          <Typography.Paragraph>
            A drawer with custom styling and behavior options.
          </Typography.Paragraph>
          <Button 
            type="primary" 
            onClick={() => setIsCustomOpen(true)}
            componentId="custom-drawer-button"
          >
            Open Custom Drawer
          </Button>
          
          <RightSlidingDrawer
            isOpen={isCustomOpen}
            onClose={() => setIsCustomOpen(false)}
            width="80%"
            animationDuration={500}
            showBackdrop
            closeOnBackdropClick
            closeOnEscape
            style={{
              backgroundColor: '#f8f9fa',
              borderLeft: '3px solid #1890ff'
            }}
          >
            <Typography.Title level={3}>Custom Styled Drawer</Typography.Title>
            <Typography.Paragraph>
              This drawer demonstrates custom styling and configuration:
            </Typography.Paragraph>
            <ul>
              <li>80% width (responsive)</li>
              <li>Slower animation (500ms)</li>
              <li>Custom background color</li>
              <li>Custom border styling</li>
            </ul>
            <hr style={{ margin: '20px 0', border: 'none', borderTop: '1px solid #e8e8e8' }} />
            <Typography.Paragraph>
              The drawer can be customized extensively to match your design system.
            </Typography.Paragraph>
          </RightSlidingDrawer>
        </div>

        {/* Custom Header Drawer */}
        <div style={cardStyle}>
          <div style={cardTitleStyle}>Custom Header Drawer</div>
          <Typography.Paragraph>
            A drawer with a completely custom header instead of the default one.
          </Typography.Paragraph>
          <Button 
            type="primary" 
            onClick={() => setIsCustomHeaderOpen(true)}
            componentId="custom-header-drawer-button"
          >
            Open Custom Header Drawer
          </Button>
          
          <RightSlidingDrawer
            isOpen={isCustomHeaderOpen}
            onClose={() => setIsCustomHeaderOpen(false)}
            width={600}
            customHeader={
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center', 
                width: '100%',
                padding: '0 20px'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <span style={{ 
                    width: '8px', 
                    height: '8px', 
                    borderRadius: '50%', 
                    backgroundColor: '#52c41a' 
                  }} />
                  <Typography.Title level={4} style={{ margin: 0, color: '#1890ff' }}>
                    🚀 Custom Header
                  </Typography.Title>
                </div>
                <Button
                  size="small"
                  onClick={() => setIsCustomHeaderOpen(false)}
                  componentId="custom-header-close-button"
                >
                  Close
                </Button>
              </div>
            }
          >
            <Typography.Title level={3}>Custom Header Example</Typography.Title>
            <Typography.Paragraph>
              This drawer demonstrates the custom header functionality:
            </Typography.Paragraph>
            <ul>
              <li>Completely custom header design</li>
              <li>Custom close button</li>
              <li>Status indicator (green dot)</li>
              <li>Emoji and custom styling</li>
            </ul>
            <hr style={{ margin: '20px 0', border: 'none', borderTop: '1px solid #e8e8e8' }} />
            <Typography.Paragraph>
              Perfect for creating branded or specialized drawer experiences.
            </Typography.Paragraph>
          </RightSlidingDrawer>
        </div>
      </div>
    </div>
  );
};

export default RightSlidingDrawerExample; 
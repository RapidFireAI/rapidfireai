import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import RightSlidingDrawer from './index';

// Mock the @databricks/design-system CloseIcon
jest.mock('@databricks/design-system', () => ({
  CloseIcon: () => <span data-testid="close-icon">×</span>,
}));

describe('RightSlidingDrawer', () => {
  const defaultProps = {
    isOpen: false,
    onClose: jest.fn(),
    children: <div>Drawer Content</div>,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders when closed', () => {
    render(<RightSlidingDrawer {...defaultProps} />);
    
    // Drawer should be rendered but positioned off-screen
    const drawer = screen.getByText('Drawer Content').closest('div');
    expect(drawer).toBeInTheDocument();
  });

  it('renders when open', () => {
    render(<RightSlidingDrawer {...defaultProps} isOpen={true} />);
    
    const drawer = screen.getByText('Drawer Content').closest('div');
    expect(drawer).toBeInTheDocument();
  });

  it('calls onClose when close button is clicked', () => {
    const onClose = jest.fn();
    render(<RightSlidingDrawer {...defaultProps} isOpen={true} onClose={onClose} />);
    
    const closeButton = screen.getByTestId('close-icon').closest('button');
    fireEvent.click(closeButton!);
    
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('renders children content', () => {
    const customContent = <div data-testid="custom-content">Custom Content</div>;
    render(<RightSlidingDrawer {...defaultProps} isOpen={true} children={customContent} />);
    
    expect(screen.getByTestId('custom-content')).toBeInTheDocument();
  });

  it('applies custom width', () => {
    render(<RightSlidingDrawer {...defaultProps} isOpen={true} width={600} />);
    
    const drawer = screen.getByText('Drawer Content').closest('div');
    expect(drawer).toHaveStyle({ width: '600px' });
  });

  it('applies custom width as string', () => {
    render(<RightSlidingDrawer {...defaultProps} isOpen={true} width="80%" />);
    
    const drawer = screen.getByText('Drawer Content').closest('div');
    expect(drawer).toHaveStyle({ width: '80%' });
  });

  it('renders without backdrop when showBackdrop is false', () => {
    render(<RightSlidingDrawer {...defaultProps} isOpen={true} showBackdrop={false} />);
    
    // Should not have backdrop-related elements
    const backdrop = document.querySelector('[data-testid="backdrop"]');
    expect(backdrop).not.toBeInTheDocument();
  });

  it('applies custom className', () => {
    const customClass = 'custom-drawer-class';
    render(<RightSlidingDrawer {...defaultProps} isOpen={true} className={customClass} />);
    
    const drawer = screen.getByText('Drawer Content').closest('div');
    expect(drawer).toHaveClass(customClass);
  });

  it('applies custom style', () => {
    const customStyle = { backgroundColor: 'red' };
    render(<RightSlidingDrawer {...defaultProps} isOpen={true} style={customStyle} />);
    
    const drawer = screen.getByText('Drawer Content').closest('div');
    expect(drawer).toHaveStyle(customStyle);
  });
}); 
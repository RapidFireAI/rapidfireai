import React, { useEffect, useRef } from 'react';
import { css, Theme } from '@emotion/react';
import { CloseIcon, useDesignSystemTheme } from '@databricks/design-system';

export interface RightSlidingDrawerProps {
  /** Whether the drawer is open */
  isOpen: boolean;
  /** Callback when the drawer should close */
  onClose: () => void;
  /** The content to render inside the drawer */
  children: React.ReactNode;
  /** Width of the drawer (default: 400px) */
  width?: number | string;
  /** Whether to show a backdrop (default: true) */
  showBackdrop?: boolean;
  /** Whether to close on backdrop click (default: true) */
  closeOnBackdropClick?: boolean;
  /** Whether to close on escape key (default: true) */
  closeOnEscape?: boolean;
  /** Custom CSS class name */
  className?: string;
  /** Custom CSS styles */
  style?: React.CSSProperties;
  /** Animation duration in milliseconds (default: 300) */
  animationDuration?: number;
  /** Whether to disable body scroll when open (default: true) */
  disableBodyScroll?: boolean;
  /** Custom header content to replace the default header */
  customHeader?: React.ReactNode;
  /** Whether to show the default header (default: true, ignored if customHeader is provided) */
  showHeader?: boolean;
}

const RightSlidingDrawer: React.FC<RightSlidingDrawerProps> = ({
  isOpen,
  onClose,
  children,
  width = 400,
  showBackdrop = true,
  closeOnBackdropClick = true,
  closeOnEscape = true,
  className,
  style,
  animationDuration = 300,
  disableBodyScroll = true,
  customHeader,
  showHeader = true,
}) => {
  const { theme } = useDesignSystemTheme();
  const drawerRef = useRef<HTMLDivElement>(null);
  const backdropRef = useRef<HTMLDivElement>(null);

  // Handle escape key
  useEffect(() => {
    if (!closeOnEscape) return;

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
    };
  }, [isOpen, onClose, closeOnEscape]);

  // Handle body scroll
  useEffect(() => {
    if (!disableBodyScroll) return;

    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }

    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen, disableBodyScroll]);

  // Handle backdrop click
  const handleBackdropClick = (event: React.MouseEvent) => {
    if (closeOnBackdropClick && event.target === backdropRef.current) {
      onClose();
    }
  };

  // Handle drawer click to prevent closing when clicking inside
  const handleDrawerClick = (event: React.MouseEvent) => {
    event.stopPropagation();
  };

  const drawerWidth = typeof width === 'number' ? `${width}px` : width;

  return (
    <>
      {/* Backdrop */}
      {showBackdrop && (
        <div
          ref={backdropRef}
          css={css`
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(0, 0, 0, 0.5);
            z-index: 1000;
            opacity: ${isOpen ? 1 : 0};
            visibility: ${isOpen ? 'visible' : 'hidden'};
            transition: all ${animationDuration}ms ease-in-out;
          `}
          onClick={handleBackdropClick}
        />
      )}

      {/* Drawer */}
      <div
        ref={drawerRef}
        css={css`
          position: fixed;
          top: 0;
          right: ${isOpen ? 0 : `-${drawerWidth}`};
          width: ${drawerWidth};
          height: 100vh;
          background-color: ${theme.colors.backgroundPrimary};
          box-shadow: -2px 0 8px rgba(0, 0, 0, 0.15);
          z-index: 1001;
          transition: right ${animationDuration}ms ease-in-out;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        `}
        className={className}
        style={style}
        onClick={handleDrawerClick}
      >
        {/* Header */}
        {(showHeader || customHeader) && (
          <div
            css={css`
              display: flex;
              justify-content: space-between;
              align-items: center;
              padding: 16px 20px;
              border-bottom: 1px solid ${theme.colors.border};
              background-color: ${theme.colors.backgroundSecondary};
              min-height: 60px;
            `}
          >
            {customHeader ? (
              customHeader
            ) : (
              <>
                <div
                  css={css`
                    font-size: 18px;
                    font-weight: 600;
                    color: ${theme.colors.textPrimary};
                  `}
                >
                  RapidFire
                </div>
                <button
                  onClick={onClose}
                  css={css`
                    background: none;
                    border: none;
                    cursor: pointer;
                    padding: 8px;
                    border-radius: 4px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: ${theme.colors.textSecondary};
                    transition: all 0.2s ease;

                    &:hover {
                      background-color: ${theme.colors.backgroundSecondary};
                      color: ${theme.colors.textPrimary};
                    }

                    &:focus {
                      outline: 2px solid ${theme.colors.primary};
                      outline-offset: 2px;
                    }
                  `}
                  aria-label="Close drawer"
                >
                  <CloseIcon />
                </button>
              </>
            )}
          </div>
        )}

        {/* Content */}
        <div
          css={css`
            flex: 1;
            overflow-y: auto;
            padding: 20px;
          `}
        >
          {children}
        </div>
      </div>
    </>
  );
};

export default RightSlidingDrawer; 
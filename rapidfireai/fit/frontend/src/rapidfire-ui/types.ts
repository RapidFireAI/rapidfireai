// Export all component types
export type { RightSlidingDrawerProps } from './components/RightSlidingDrawer';

// Common types that might be used across components
export interface BaseComponentProps {
  /** Custom CSS class name */
  className?: string;
  /** Custom CSS styles */
  style?: React.CSSProperties;
  /** Whether the component is disabled */
  disabled?: boolean;
  /** Whether the component is loading */
  loading?: boolean;
}

export interface AnimationProps {
  /** Animation duration in milliseconds */
  animationDuration?: number;
  /** Animation easing function */
  animationEasing?: string;
  /** Whether animations are enabled */
  enableAnimations?: boolean;
}

export interface AccessibilityProps {
  /** ARIA label for screen readers */
  ariaLabel?: string;
  /** ARIA described by element ID */
  ariaDescribedBy?: string;
  /** ARIA controls element ID */
  ariaControls?: string;
  /** Whether the component should be focusable */
  focusable?: boolean;
  /** Tab index for keyboard navigation */
  tabIndex?: number;
}

export interface DrawerProps {
  /** Whether the drawer is open */
  isOpen: boolean;
  /** Callback when the drawer should close */
  onClose: () => void;
  /** Width of the drawer */
  width?: number | string;
  /** Whether to show a backdrop */
  showBackdrop?: boolean;
  /** Whether to close on backdrop click */
  closeOnBackdropClick?: boolean;
  /** Whether to close on escape key */
  closeOnEscape?: boolean;
  /** Animation duration in milliseconds */
  animationDuration?: number;
  /** Whether to disable body scroll when open */
  disableBodyScroll?: boolean;
} 
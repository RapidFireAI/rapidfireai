import { useDesignSystemTheme } from '@databricks/design-system';

export const InteractiveControllerIcon = () => {
  const { theme } = useDesignSystemTheme();
  
  // Use theme-aware colors: white for dark mode, dark gray for light mode
  const iconColor = theme.colors.textPrimary;
  
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
      {/* First slider */}
      <line x1="6" y1="6" x2="6" y2="18" stroke={iconColor} strokeWidth="2" />
      <circle cx="6" cy="12" r="3" fill={iconColor} />
      
      {/* Second slider */}
      <line x1="12" y1="6" x2="12" y2="18" stroke={iconColor} strokeWidth="2" />
      <circle cx="12" cy="8" r="3" fill={iconColor} />
      
      {/* Third slider */}
      <line x1="18" y1="6" x2="18" y2="18" stroke={iconColor} strokeWidth="2" />
      <circle cx="18" cy="15" r="3" fill={iconColor} />
    </svg>
  );
};

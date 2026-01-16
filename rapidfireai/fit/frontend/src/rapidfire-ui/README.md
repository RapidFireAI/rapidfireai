# RapidFire UI Components

A collection of reusable UI components built specifically for RapidFire applications, designed to complement the existing `@databricks/design-system` components.

## Components

### RightSlidingDrawer

A right-side sliding drawer component with smooth animations and accessibility features.

#### Features

- **Smooth sliding animation** from right to left
- **Backdrop support** with click-to-close functionality
- **Keyboard navigation** (ESC key to close)
- **Body scroll management** (prevents background scrolling when open)
- **Customizable width** and styling
- **Accessibility features** (ARIA labels, focus management)
- **TypeScript support** with full type definitions

#### Usage

```tsx
import { RightSlidingDrawer } from 'rapidfire-ui';
import { useState } from 'react';

const MyComponent = () => {
  const [isOpen, setIsOpen] = useState(false);

  const customHeader = (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
      <h3>Custom Header</h3>
      <button onClick={() => setIsOpen(false)}>Close</button>
    </div>
  );

  return (
    <RightSlidingDrawer
      isOpen={isOpen}
      onClose={() => setIsOpen(false)}
      customHeader={customHeader}
      width={500}
    >
      <h2>Your Content Here</h2>
      <p>Any React content can go inside!</p>
    </RightSlidingDrawer>
  );
};
```

#### Custom Headers

You can customize the drawer header by providing a `customHeader` prop:

#### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `isOpen` | `boolean` | **required** | Whether the drawer is open |
| `onClose` | `() => void` | **required** | Callback when the drawer should close |
| `children` | `React.ReactNode` | **required** | The content to render inside the drawer |
| `width` | `number \| string` | `400` | Width of the drawer (px or CSS value) |
| `showBackdrop` | `boolean` | `true` | Whether to show a backdrop |
| `closeOnBackdropClick` | `boolean` | `true` | Whether to close on backdrop click |
| `closeOnEscape` | `boolean` | `true` | Whether to close on escape key |
| `className` | `string` | - | Custom CSS class name |
| `style` | `React.CSSProperties` | - | Custom CSS styles |
| `animationDuration` | `number` | `300` | Animation duration in milliseconds |
| `disableBodyScroll` | `boolean` | `true` | Whether to disable body scroll when open |
| `customHeader` | `React.ReactNode` | - | Custom header content to replace the default header |
| `showHeader` | `boolean` | `true` | Whether to show the default header (ignored if customHeader is provided) |

#### Styling

The component uses Emotion CSS-in-JS and follows RapidFire design patterns. You can customize the appearance using:

- `className` prop for additional CSS classes
- `style` prop for inline styles
- CSS custom properties for theming
- Emotion's `css` prop for component-level overrides

#### Accessibility

- **Keyboard navigation**: ESC key closes the drawer
- **Focus management**: Focus is trapped within the drawer when open
- **ARIA labels**: Proper labeling for screen readers
- **Semantic HTML**: Uses appropriate HTML elements and attributes

#### Browser Support

- Modern browsers with CSS transitions support
- IE11+ (with polyfills for CSS transitions)
- Mobile browsers with touch support

## Examples

The `RightSlidingDrawerExample` component demonstrates three different use cases:

1. **Basic Drawer**: Default settings with 400px width
2. **Wide Drawer**: 600px width for content that needs more space
3. **Custom Styled Drawer**: 80% width with custom styling and slower animation

To use the examples:

```tsx
import { RightSlidingDrawerExample } from 'rapidfire-ui';

// In your component
<RightSlidingDrawerExample />
```

## Development

### Adding New Components

1. Create a new folder in `components/` with your component name
2. Create an `index.tsx` file with your component
3. Export the component and its types from the main `index.ts`
4. Add documentation to this README
5. Include TypeScript interfaces for all props

### Component Guidelines

- Use TypeScript for all components
- Follow the existing naming conventions
- Use Emotion for styling
- Include proper accessibility features
- Provide comprehensive prop interfaces
- Add JSDoc comments for complex props
- Test with different screen sizes and devices

### Testing

Components should be tested for:
- Functionality across different browsers
- Accessibility compliance
- Responsive design
- Keyboard navigation
- Screen reader compatibility

## Dependencies

- React 16.8+
- TypeScript 4.0+
- Emotion (CSS-in-JS)
- @databricks/design-system (for icons and base components)

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure to import from the correct path
   ```tsx
   // Correct
   import RightSlidingDrawer from 'rapidfire-ui/components/RightSlidingDrawer';
   
   // Or use the main export
   import { RightSlidingDrawer } from 'rapidfire-ui';
   ```

2. **Component not found**: Ensure the component is properly exported from the main index file

3. **Styling issues**: Check that Emotion is properly configured in your project

4. **TypeScript errors**: Verify that all required props are provided and types match 
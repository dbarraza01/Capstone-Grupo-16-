---
name: Clinical Intelligence System
colors:
  surface: '#f9f9ff'
  surface-dim: '#cfdaf1'
  surface-bright: '#f9f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f0f3ff'
  surface-container: '#e7eeff'
  surface-container-high: '#dee8ff'
  surface-container-highest: '#d8e3fa'
  on-surface: '#111c2c'
  on-surface-variant: '#42474f'
  inverse-surface: '#263142'
  inverse-on-surface: '#ebf1ff'
  outline: '#727780'
  outline-variant: '#c2c7d1'
  surface-tint: '#2d6197'
  primary: '#00355f'
  on-primary: '#ffffff'
  primary-container: '#0f4c81'
  on-primary-container: '#8ebdf9'
  inverse-primary: '#a0c9ff'
  secondary: '#006a61'
  on-secondary: '#ffffff'
  secondary-container: '#6af5e5'
  on-secondary-container: '#006f66'
  tertiary: '#532800'
  on-tertiary: '#ffffff'
  tertiary-container: '#743b00'
  on-tertiary-container: '#f9a767'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#d2e4ff'
  primary-fixed-dim: '#a0c9ff'
  on-primary-fixed: '#001c37'
  on-primary-fixed-variant: '#07497d'
  secondary-fixed: '#6df8e7'
  secondary-fixed-dim: '#4bdbcb'
  on-secondary-fixed: '#00201d'
  on-secondary-fixed-variant: '#005049'
  tertiary-fixed: '#ffdcc4'
  tertiary-fixed-dim: '#ffb780'
  on-tertiary-fixed: '#2f1400'
  on-tertiary-fixed-variant: '#6f3800'
  background: '#f9f9ff'
  on-background: '#111c2c'
  surface-variant: '#d8e3fa'
typography:
  display-lg:
    fontFamily: Geist
    fontSize: 48px
    fontWeight: '600'
    lineHeight: '1.1'
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Geist
    fontSize: 32px
    fontWeight: '600'
    lineHeight: '1.2'
    letterSpacing: -0.01em
  headline-lg-mobile:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.2'
  headline-md:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '500'
    lineHeight: '1.3'
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: '1.6'
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.5'
  label-md:
    fontFamily: Geist
    fontSize: 14px
    fontWeight: '500'
    lineHeight: '1.4'
    letterSpacing: 0.01em
  label-sm:
    fontFamily: Geist
    fontSize: 12px
    fontWeight: '600'
    lineHeight: '1.2'
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  unit: 4px
  container-max: 1440px
  gutter: 24px
  margin-mobile: 16px
  margin-desktop: 40px
  stack-sm: 8px
  stack-md: 16px
  stack-lg: 32px
---

## Brand & Style

The design system is engineered for high-stakes medical decision-making where clarity, precision, and technological sophistication are paramount. The brand personality is "Clinical Excellence through Intelligence"—it is authoritative yet approachable, minimizing cognitive load for practitioners while maintaining a premium, tech-forward aesthetic.

The visual style blends **Minimalism** with **Glassmorphism** and **Tonal Layering**, heavily influenced by modern developer-centric tools. It prioritizes information density without clutter, utilizing generous white space and a "content-first" hierarchy. The result is an interface that feels more like a sophisticated diagnostic tool and less like a legacy administrative database.

## Colors

The palette is anchored in a neutral, cool-toned environment to reduce eye strain during long clinical shifts.

*   **Primary (Clinical Blue):** Used for primary actions, active states, and authoritative branding.
*   **Accent (Turquoise):** Reserved for AI-driven insights, data highlights, and suggested actions to differentiate "System Intelligence" from "Standard Operations."
*   **Surface:** A hierarchy of off-whites and cool grays (#F8FAFC to White) creates depth without the use of heavy lines.
*   **Semantic:** High-saturation reds and greens are used sparingly for immediate status communication (Critical Alerts vs. Successful Validation).

## Typography

This design system utilizes a dual-font strategy. **Geist** provides a technical, precise feel for headings and UI labels, emphasizing the "intelligence" aspect of the system. **Inter** is used for all body text and patient data to ensure maximum legibility and comfort for long-form reading.

Typography follows a strict scale to maintain a clear information hierarchy. Data points should always use `label-md` or `body-md` for clarity, while AI-generated summaries utilize `body-lg` to prioritize synthesis over raw data entry.

## Layout & Spacing

The system employs a **Fluid-Fixed hybrid grid**. The main content area lives within a 1440px max-width container, centered on the screen, while sidebars and navigation panels remain anchored to the viewport edges.

*   **Rhythm:** An 8px base grid governs all spacing. 
*   **Density:** While the design is minimalist, it allows for high-density "Data Modes" where spacing collapses to 4px units for complex medical charts, provided the visual weight is balanced by increased whitespace in the surrounding container.
*   **Breakpoints:**
    *   *Mobile (<768px):* Single column, 16px margins, bottom-anchored primary actions.
    *   *Tablet (768px - 1024px):* 8-column grid, collapsed sidebars.
    *   *Desktop (>1024px):* 12-column grid, persistent navigation, 40px margins.

## Elevation & Depth

Hierarchy is established through **Tonal Layering** and **Ambient Shadows**. Instead of heavy borders, the design system uses subtle shifts in background color and soft shadows to separate the interface into three planes:

1.  **Floor (Level 0):** The base background (#F8FAFC).
2.  **Card/Surface (Level 1):** Pure white (#FFFFFF) with a very soft, diffused shadow (0px 4px 20px rgba(0,0,0,0.04)). Use a 1px border (#E2E8F0) for crispness.
3.  **Floating/Overlay (Level 2):** High elevation (0px 10px 40px rgba(0,0,0,0.08)) for modals and tooltips, utilizing a 20px backdrop blur (Glassmorphism) to maintain context.

AI-generated components should use a subtle inner-glow in the Accent color (#00B8A9) to signify their dynamic nature.

## Shapes

The design system uses a **Rounded** (0.5rem base) shape language to feel modern and accessible.

*   **Standard Components:** Buttons, inputs, and cards use an 8px (0.5rem) radius.
*   **Container Elements:** Large sections and dashboard widgets use a 16px (1rem) radius.
*   **Pill Elements:** Status tags, AI chips, and search bars use a fully rounded (999px) radius to provide visual variety and signify "interactive objects."

## Components

### Buttons & Inputs
*   **Primary Button:** Clinical Blue background, white text, 8px radius. Subtle scale-down effect on click (0.98x).
*   **AI Action Button:** Turquoise background with a subtle gradient. Uses a "sparkle" icon prefix.
*   **Input Fields:** Ghost-style borders (#E2E8F0) that transition to Clinical Blue on focus with a 3px soft outer glow.

### Cards & Surfaces
*   **Patient Cards:** Level 1 elevation. Metadata is displayed using `label-sm` in Neutral Gray.
*   **Intelligence Insights:** These cards feature a 2px left-border highlight in Turquoise to indicate AI-driven content.

### Medical Specifics
*   **Vital Indicators:** Micro-sparklines integrated into list items to show 24h trends without requiring a full chart view.
*   **Status Chips:** Use low-saturation background tints with high-saturation text for high legibility (e.g., Success: Light green bg, dark green text).
*   **Data Lists:** No horizontal borders between items; use vertical spacing and hover states (subtle #F1F5F9 background) to define rows.

### Micro-interactions
Transitions between dashboard views must be fluid (300ms ease-out). AI-loading states should use a shimmering "pulse" rather than a spinning loader to maintain the "premium tech" feel.
# Sphana Documentation

Modern React-based documentation for the Sphana Neural RAG Database project.

## Tech Stack

- **React 18** with TypeScript
- **Vite 5** for fast development and optimized builds
- **Tailwind CSS** for styling
- **React Router** for client-side routing
- **React Markdown** for rendering documentation
- **Fuse.js** for search functionality

## Getting Started

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

The documentation site will be available at `http://localhost:5173`

### Build

```bash
npm run build
```

This creates an optimized production build in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## Project Structure

- `src/` - React application source code
  - `components/` - Reusable React components
  - `hooks/` - Custom React hooks
  - `types/` - TypeScript type definitions
  - `utils/` - Utility functions
- `public/content/` - Markdown documentation files
- `tailwind.config.js` - Tailwind CSS configuration
- `vite.config.ts` - Vite configuration

## Adding Documentation

To add or update documentation:

1. Edit Markdown files in `public/content/`
2. Update `public/content/navigation.json` if adding new pages
3. No rebuild required - changes are reflected immediately in development mode

## Features

- 📱 Fully responsive design
- 🌙 Dark mode support
- 🔍 Full-text search with fuzzy matching
- 📝 Markdown rendering with syntax highlighting
- ⚡ Fast page loads with code splitting
- ♿ Accessible (WCAG compliant)


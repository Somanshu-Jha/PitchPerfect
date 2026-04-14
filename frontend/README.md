# PitchPerfect Frontend

The PitchPerfect frontend is a React + Vite application that provides authentication, interview recording, evaluation results, and history dashboards for the AI interview evaluation platform.

## Features

- Login and signup flows
- Recording/upload experience for interview audio
- Result summaries with coaching feedback
- History and progress tracking
- Developer controls for strictness configuration

## Getting Started

### Prerequisites
- Node.js 18+

### Install & Run
```bash
npm install
npm run dev
```

### Lint & Build
```bash
npm run lint
npm run build
```

## API Configuration

The frontend currently calls the backend at `http://localhost:8000`. Update the hard-coded base URLs in the source if you need a different host.

## Project Structure

```
src/components/   UI components and screens
src/lib/          Shared utilities
src/App.tsx       Application shell
```

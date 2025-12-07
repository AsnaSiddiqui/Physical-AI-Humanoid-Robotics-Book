# Quickstart Guide: AI Agents, MCP & Docusaurus Project

## Prerequisites

- Node.js (v18 or higher)
- npm or yarn package manager
- Git
- Access to Claude Code and MCP Server

## Initial Setup

### 1. Verify Docusaurus Installation
First, ensure Docusaurus is available in your environment:
```bash
node -v
npm -v
```

### 2. Create Project Structure
The project must be structured with all Docusaurus-related files inside the `frontend/` directory:
```bash
mkdir -p frontend/docs
mkdir -p frontend/src
mkdir -p frontend/static
```

### 3. Initialize Docusaurus in frontend/
```bash
cd frontend
npm init -y
```

### 4. Verify MCP Server Connection
Before proceeding with any Docusaurus work, verify the MCP Server:
```bash
# Check if MCP server is accessible
# Implementation will depend on your specific MCP setup
```

## Project Initialization Checklist

- [ ] Docusaurus is installed and accessible
- [ ] frontend/ directory exists
- [ ] MCP Server connection verified
- [ ] Node.js and npm are properly configured
- [ ] Git repository is initialized

## Running the Documentation Site

### Development Mode
```bash
cd frontend
npm start
```

### Build for Production
```bash
cd frontend
npm run build
```

### Serve Production Build
```bash
npm run serve
```

## AI Agent Workflow

### 1. Research Phase
- Use Research Expert sub-agent to gather information
- Apply Content Architect skill to structure information
- Create research packets for each topic

### 2. Writing Phase
- Use Technical Writer sub-agent for content creation
- Apply Style & Consistency Editor for formatting
- Validate with Quality Validator

### 3. Publishing Phase
- Optimize with SEO & Docsite Optimization Agent
- Publish with Publishing & Versioning Agent
- Use Documentation Website Builder and SEO Formatter skills

## Common Commands

### Docusaurus Commands (run from frontend/)
- `npm run start` - Start development server
- `npm run build` - Build static site
- `npm run serve` - Serve production build
- `npm run swizzle` - Customize components
- `npm run deploy` - Deploy to GitHub Pages

### File Locations
- All documentation content: `frontend/docs/`
- Configuration: `frontend/docusaurus.config.js`
- Navigation: `frontend/sidebars.js`
- Custom components: `frontend/src/`
- Static assets: `frontend/static/`

## Troubleshooting

### Common Issues

1. **Docusaurus not found**
   - Ensure you're running commands from the frontend/ directory
   - Verify Docusaurus is properly installed

2. **MCP Server connection issues**
   - Check server status
   - Verify network connectivity
   - Confirm authentication credentials

3. **Frontend directory missing**
   - Create the frontend/ directory as specified
   - Ensure all Docusaurus files are placed inside it
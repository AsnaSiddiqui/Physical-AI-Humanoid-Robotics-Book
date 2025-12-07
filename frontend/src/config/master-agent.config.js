/**
 * AI Documentation Architect Master Agent Configuration
 * Coordinates all sub-agents and skills for the documentation project
 */

const masterAgentConfig = {
  name: 'AI Documentation Architect',
  version: '1.0.0',
  description: 'Master agent that controls workflows, calls sub-agents, executes skills, ensures quality, and builds the final Docusaurus site',

  // Sub-agents configuration
  subAgents: {
    'research-expert': {
      enabled: true,
      priority: 1,
      capabilities: [
        'collect-core-research',
        'extract-key-insights',
        'build-book-architecture'
      ]
    },
    'technical-writer': {
      enabled: true,
      priority: 2,
      capabilities: [
        'write-chapter-drafts',
        'create-docusaurus-content'
      ]
    },
    'editor-consistency-reviewer': {
      enabled: true,
      priority: 3,
      capabilities: [
        'style-consistency-check',
        'formatting-validation'
      ]
    },
    'quality-validator': {
      enabled: true,
      priority: 4,
      capabilities: [
        'technical-accuracy-verification',
        'content-quality-assessment'
      ]
    },
    'seo-docsite-optimization-agent': {
      enabled: true,
      priority: 5,
      capabilities: [
        'seo-optimization',
        'meta-tag-generation'
      ]
    },
    'publishing-versioning-agent': {
      enabled: true,
      priority: 6,
      capabilities: [
        'site-publishing',
        'version-management'
      ]
    }
  },

  // Skills configuration
  skills: {
    'content-architect': {
      enabled: true,
      functions: [
        'structure-content',
        'create-outlines',
        'define-information-architecture'
      ]
    },
    'style-consistency-editor': {
      enabled: true,
      functions: [
        'format-content',
        'ensure-tone-consistency',
        'apply-style-guidelines'
      ]
    },
    'documentation-website-builder': {
      enabled: true,
      functions: [
        'create-frontend-structure',
        'manage-docusaurus-config',
        'build-site-structure'
      ]
    },
    'seo-formatter': {
      enabled: true,
      functions: [
        'optimize-meta-tags',
        'generate-seo-content',
        'create-search-friendly-content'
      ]
    },
    'github-deploy-manager': {
      enabled: true,
      functions: [
        'deploy-to-github-pages',
        'manage-deployment-workflow'
      ]
    }
  },

  // Workflow configuration
  workflow: {
    phases: [
      'research-foundation',
      'section-level-research',
      'writing-editing',
      'final-build-publishing'
    ],
    checkpoints: {
      'research-foundation': 'checkpoint-1-required',
      'section-level-research': 'checkpoint-2-required',
      'writing-editing': 'checkpoint-3-required'
    }
  },

  // MCP Server integration
  mcpIntegration: {
    verifyBeforeDocusaurusWork: true,
    projectStructureCheck: true
  },

  // Directory configuration
  directoryRules: {
    docusaurusFilesLocation: 'frontend/',
    enforceFrontendDirectory: true,
    docusaurusAlreadyInstalled: true
  }
};

module.exports = masterAgentConfig;
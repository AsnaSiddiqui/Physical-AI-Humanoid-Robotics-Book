/**
 * Skills Configuration for AI Documentation Project
 * Configuration for Content Architect, Style & Consistency Editor, etc.
 */

const skillsConfig = {
  'content-architect': {
    name: 'Content Architect',
    description: 'Structure, outlines, information architecture',
    functions: [
      'structure-content',
      'create-outlines',
      'define-information-architecture',
      'build-high-level-book-architecture',
      'organize-research-into-structured-content'
    ],
    priority: 1,
    enabled: true
  },

  'style-consistency-editor': {
    name: 'Style & Consistency Editor',
    description: 'Tone, formatting, clarity',
    functions: [
      'apply-style-consistency-edits',
      'ensure-tone-consistency',
      'format-content-properly',
      'apply-clarity-improvements',
      'apply-to-ai-documentation-architect-chapter',
      'apply-to-sub-agent-design-chapter',
      'apply-to-skills-integration-chapter'
    ],
    priority: 2,
    enabled: true
  },

  'documentation-website-builder': {
    name: 'Documentation Website Builder',
    description: 'All work must go inside frontend/',
    functions: [
      'create-frontend-directory-structure',
      'build-docusaurus-site',
      'manage-frontend-directory-structure',
      'create-initial-docs-directory-structure'
    ],
    priority: 3,
    enabled: true,
    constraints: {
      allWorkInFrontend: true
    }
  },

  'seo-formatter': {
    name: 'SEO Formatter',
    description: 'SEO optimization',
    functions: [
      'apply-seo-optimization',
      'format-for-search-engines',
      'optimize-meta-tags',
      'create-seo-friendly-content'
    ],
    priority: 4,
    enabled: true
  },

  'github-deploy-manager': {
    name: 'GitHub Deploy Manager',
    description: 'Deployment operations',
    functions: [
      'configure-deployment-process',
      'manage-github-deployment',
      'handle-deployment-operations'
    ],
    priority: 5,
    enabled: true
  }
};

module.exports = skillsConfig;
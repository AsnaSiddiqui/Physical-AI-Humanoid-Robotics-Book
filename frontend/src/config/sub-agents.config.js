/**
 * Sub-Agent Configuration for AI Documentation Project
 * Configuration for Research Expert, Technical Writer, etc.
 */

const subAgentsConfig = {
  'research-expert': {
    name: 'Research Expert',
    description: 'Collects core research, extracts key insights, builds book architecture',
    functions: [
      'collect-core-research-from-pdf',
      'extract-key-insights',
      'build-high-level-book-architecture',
      'research-each-chapter-topic',
      'create-summaries-for-every-section',
      'add-advanced-topics',
      'assemble-section-research-packets'
    ],
    priority: 1,
    enabled: true
  },

  'technical-writer': {
    name: 'Technical Writer',
    description: 'Writes full chapter drafts in Docusaurus format',
    functions: [
      'write-full-chapter-drafts',
      'create-docusaurus-compatible-content',
      'draft-ai-documentation-architect-chapter',
      'draft-sub-agent-design-chapter',
      'draft-skills-integration-chapter'
    ],
    priority: 2,
    enabled: true
  },

  'editor-consistency-reviewer': {
    name: 'Editor & Consistency Reviewer',
    description: 'Ensures style and formatting consistency',
    functions: [
      'apply-style-consistency-edits',
      'validate-formatting-standards',
      'ensure-tone-consistency',
      'apply-style-consistency-editor-skill'
    ],
    priority: 3,
    enabled: true
  },

  'quality-validator': {
    name: 'Quality Validator',
    description: 'Validates technical accuracy',
    functions: [
      'validate-technical-accuracy',
      'perform-final-review',
      'validate-content-quality'
    ],
    priority: 4,
    enabled: true
  },

  'seo-docsite-optimization-agent': {
    name: 'SEO & Docsite Optimization Agent',
    description: 'Applies proper SEO optimization',
    functions: [
      'apply-seo-optimization',
      'optimize-site-content',
      'optimize-meta-tags-descriptions'
    ],
    priority: 5,
    enabled: true
  },

  'publishing-versioning-agent': {
    name: 'Publishing & Versioning Agent',
    description: 'Handles final publishing and versioning',
    functions: [
      'handle-publishing-versioning',
      'prepare-site-for-deployment',
      'manage-version-control'
    ],
    priority: 6,
    enabled: true
  }
};

module.exports = subAgentsConfig;
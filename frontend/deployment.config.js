/**
 * Deployment Configuration for AI Agents, MCP & Docusaurus Guide
 * Configuration for GitHub Pages deployment
 */

const deploymentConfig = {
  // GitHub Pages deployment settings
  githubPages: {
    enabled: true,
    organizationName: 'your-organization', // Usually your GitHub org/user name
    projectName: 'ai-agents-mcp-docusaurus', // Usually your repo name
    deploymentBranch: 'gh-pages',
    buildCommand: 'npm run build',
    outputDirectory: 'build',
    cname: null, // Set to your custom domain if applicable
  },

  // Deployment process
  deploymentProcess: {
    preBuildSteps: [
      'npm install',
      'dependency validation',
      'configuration validation'
    ],
    buildStep: 'npm run build',
    postBuildSteps: [
      'build validation',
      'link checking',
      'content accuracy verification'
    ],
    deploymentStep: 'gh-pages -d build',
    verificationSteps: [
      'site accessibility check',
      'content display validation',
      'navigation functionality test'
    ]
  },

  // Deployment triggers
  triggers: {
    onPushTo: ['main', 'master'],
    onPullRequest: false,
    manualDeploy: true
  },

  // Environment configuration
  environments: {
    development: {
      url: 'http://localhost:3000',
      buildCommand: 'npm run start'
    },
    staging: {
      url: 'https://staging.your-site.com',
      buildCommand: 'npm run build',
      outputDirectory: 'build-staging'
    },
    production: {
      url: 'https://your-organization.github.io/ai-agents-mcp-docusaurus',
      buildCommand: 'npm run build',
      outputDirectory: 'build',
      cname: null
    }
  },

  // GitHub integration
  githubIntegration: {
    token: 'GITHUB_TOKEN', // Reference to environment variable
    repository: 'your-organization/ai-agents-mcp-docusaurus',
    branch: 'main',
    pagesSource: {
      branch: 'gh-pages',
      path: '/'
    }
  },

  // Performance monitoring
  performance: {
    budget: {
      maxBundleSize: '5MB',
      maxInitialSize: '2MB',
      warnOnExceed: true
    },
    metrics: [
      'loadTime',
      'firstContentfulPaint',
      'largestContentfulPaint',
      'cumulativeLayoutShift'
    ]
  },

  // SEO verification
  seoVerification: {
    metaTags: true,
    sitemap: true,
    structuredData: false,
    performance: true
  },

  // Content verification
  contentVerification: {
    brokenLinks: true,
    imageOptimization: true,
    accessibility: true,
    mobileResponsive: true
  }
};

module.exports = deploymentConfig;
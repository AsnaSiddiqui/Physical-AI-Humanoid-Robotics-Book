# Data Model for Physical AI & Humanoid Robotics Book

Since this project is primarily a documentation/book creation project rather than a traditional software application, the "data model" refers to the structure and relationships of content elements rather than database schemas.

## Content Entities

### Chapter
- **id**: Unique identifier for the chapter
- **title**: The title of the chapter
- **description**: SEO-friendly description
- **slug**: URL-friendly identifier
- **tags**: Array of relevant tags
- **module_id**: Reference to the parent module
- **content**: The main content of the chapter in Markdown format
- **created_date**: Date the chapter was created
- **last_modified**: Date the chapter was last updated
- **word_count**: Number of words in the chapter
- **status**: Draft, Review, Published

### Module
- **id**: Unique identifier for the module
- **title**: The title of the module
- **description**: Brief description of the module
- **chapters**: Array of chapter IDs belonging to this module
- **order**: Order in which the module appears in the book

### Source
- **id**: Unique identifier for the source
- **title**: Title of the source
- **author**: Author(s) of the source
- **date**: Publication date
- **url**: URL to the source
- **type**: Type of source (academic, documentation, tutorial, book, etc.)
- **relevance**: Description of how the source is relevant to the content
- **module_id**: Reference to the module this source is associated with
- **chapter_id**: Reference to the chapter this source is associated with

### User
- **id**: Unique identifier for the user
- **username**: User's chosen username
- **email**: User's email address
- **role**: User's role (reader, contributor, admin)
- **preferences**: User preferences including background (software/hardware)
- **language**: Preferred language (for translation features)

### Audit Log
- **id**: Unique identifier for the audit entry
- **chapter_id**: Reference to the chapter being audited
- **generated_by**: Subagent that generated the content
- **checked_by**: Subagent or skill that reviewed the content
- **quality_status**: Approval status from quality-validator
- **timestamp**: When the audit entry was created
- **changes**: Description of any changes made

## Relationships

- A Module contains many Chapters
- A Chapter belongs to one Module
- A Chapter can reference many Sources
- A Module can reference many Sources
- A User can access many Chapters
- An Audit Log entry references one Chapter

## Content Validation Rules

- Each Chapter must have a unique slug within its Module
- Each Chapter must have 1-10 Sources associated with it
- Chapter word count must be between 800-1500 words
- All content must be original (no direct copying)
- All claims must be supported by sources
- Front matter must include all required fields (title, description, slug, tags)

## State Transitions

### Chapter States
- Draft → Review (when initial writing is complete)
- Review → Revision (if changes are requested)
- Review → Published (if approved)
- Revision → Review (after changes are made)
- Published → Revision (if updates are needed)

## Content Structure

The content model follows a hierarchical structure:
- Book (entire textbook)
  - Module (major sections, e.g., Module 1: ROS 2)
    - Chapter (individual lessons)
      - Section (subdivisions within chapters)
        - Content Block (specific pieces of content: text, code, examples)
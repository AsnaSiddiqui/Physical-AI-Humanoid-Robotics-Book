# Quickstart Guide for Physical AI & Humanoid Robotics Book

This guide will help you get started with the Physical AI & Humanoid Robotics textbook project.

## Prerequisites

- Node.js (v18 or higher)
- npm or yarn package manager
- Python 3.8 or higher
- ROS 2 (Rolling Ridley or Humble Hawksbill)
- Git

## Setting Up the Development Environment

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install Docusaurus dependencies:
   ```bash
   npm install
   ```

3. Verify ROS 2 installation:
   ```bash
   source /opt/ros/humble/setup.bash  # or your ROS 2 distribution
   ros2 --version
   ```

## Running the Documentation Site Locally

1. Start the development server:
   ```bash
   npm run start
   ```

2. Open your browser to `http://localhost:3000` to view the site.

## Building the Site

To build the static site for production:

```bash
npm run build
```

The built site will be available in the `build` directory.

## Project Structure

- `/docs` - Contains all the textbook content in Markdown format
- `/src` - Custom React components and styling
- `/static` - Static assets like images
- `sidebar.js` - Navigation sidebar configuration
- `docusaurus.config.js` - Main Docusaurus configuration

## Contributing to the Textbook

1. Each chapter follows the format:
   ```
   ---
   title: "Chapter Title"
   description: "SEO-friendly description"
   slug: /module-x/chapter-y
   tags: [ROS2, Gazebo, Isaac, Humanoid-Robotics, Physical-AI]
   ---
   ```

2. Add new chapters to the appropriate module directory in `/docs`.

3. Update `sidebar.js` to include new chapters in the navigation.
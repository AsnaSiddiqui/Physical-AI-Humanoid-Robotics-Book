const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 3000;

// Serve static files from the build directory (or docs for development)
app.use(express.static('.'));

// Handle Markdown files by serving them as text
app.get(/\.md$/, (req, res) => {
  const filePath = path.join(__dirname, req.path);
  fs.readFile(filePath, 'utf8', (err, data) => {
    if (err) {
      res.status(404).send('File not found');
      return;
    }
    res.setHeader('Content-Type', 'text/plain');
    res.send(data);
  });
});

// Serve the main page
app.get('/', (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>Physical AI & Humanoid Robotics Book</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #2c3e50; }
        ul { line-height: 1.8; }
        a { color: #3498db; text-decoration: none; }
        a:hover { text-decoration: underline; }
      </style>
    </head>
    <body>
      <h1>Physical AI & Humanoid Robotics Book</h1>
      <p>This is the development server for the Physical AI & Humanoid Robotics book.</p>
      <h2>Available Documentation:</h2>
      <ul>
        <li><a href="/docs/intro.md">Introduction to Physical AI & Humanoid Robotics</a></li>
        <li><a href="/docs/book-architecture.md">Book Architecture: Physical AI & Humanoid Robotics</a></li>
        <li><a href="/docs/getting-started.md">Getting Started with Physical AI & Humanoid Robotics</a></li>
        <li><a href="/docs/setup.md">Project Setup Guide for Physical AI & Humanoid Robotics</a></li>
        <li><a href="/docs/01-intro-ros2.md">Module 1: The Robotic Nervous System (ROS 2)</a></li>
        <li><a href="/docs/02-ros2-architecture.md">ROS 2 Communication Patterns</a></li>
        <li><a href="/docs/03-urdf-modeling.md">URDF for Humanoid Robot Modeling</a></li>
      </ul>
      <p>All documentation files are available in the <code>/docs/</code> directory.</p>
    </body>
    </html>
  `);
});

app.listen(PORT, () => {
  console.log(`Development server running at http://localhost:${PORT}`);
  console.log('Documentation available at the links above');
});
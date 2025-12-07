module.exports = {
  bookSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: [
        'intro',
        'prerequisites',
        'setup'
      ],
    },
    {
      type: 'category',
      label: 'Module 1 – The Robotic Nervous System (ROS 2)',
      items: [
        'module-1-ros2/01-intro-ros2',
        'module-1-ros2/02-ros2-architecture',
        'module-1-ros2/03-rclpy',
        'module-1-ros2/04-urdf-xacro',
        'module-1-ros2/05-ros2-control'
      ],
    },
    {
      type: 'category',
      label: 'Module 2 – The Digital Twin (Gazebo & Unity)',
      items: [
        'module-2-digital-twin/06-digital-twin',
        'module-2-digital-twin/07-gazebo-basics',
        'module-2-digital-twin/08-urdf-sdf-sim',
        'module-2-digital-twin/09-sensor-simulation',
        'module-2-digital-twin/10-unity-visualization'
      ],
    },
    {
      type: 'category',
      label: 'Module 3 – The AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'module-3-isaac/11-isaac-intro',
        'module-3-isaac/12-isaac-ros-perception',
        'module-3-isaac/13-nav2',
        'module-3-isaac/14-manipulation',
        'module-3-isaac/15-sim-to-real'
      ],
    },
    {
      type: 'category',
      label: 'Module 4 – Vision-Language-Action (VLA)',
      items: [
        'module-4-vla/16-voice-to-action',
        'module-4-vla/17-llm-planning',
        'module-4-vla/18-vla-systems',
        'module-4-vla/19-conversational-robotics',
        'module-4-vla/20-capstone'
      ],
    },
    {
      type: 'category',
      label: 'Appendices',
      items: [
        'appendices/troubleshooting',
        'appendices/glossary',
        'appendices/references'
      ],
    }
  ],
};
// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
    tutorialSidebar: [
    {
      type: 'doc',
      id: 'intro',
    },
    {
      type: 'doc',
      id: 'book-architecture',
    },
    {
      type: 'doc',
      id: 'getting-started',
    },
    {
      type: 'doc',
      id: 'setup',
    },
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        {
          type: 'doc',
          id: 'module-1-ros2/01-intro-ros2',
        },
        {
          type: 'doc',
          id: 'module-1-ros2/02-ros2-architecture',
        },
        {
          type: 'doc',
          id: 'module-1-ros2/03-urdf-modeling',
        },
        {
          type: 'doc',
          id: 'module-1-ros2/04-simulation-environments',
        },
        {
          type: 'doc',
          id: 'module-1-ros2/module-1-summary',
        }
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        {
          type: 'doc',
          id: 'module-2-gazebo-unity/module-2-overview',
        },
        {
          type: 'doc',
          id: 'module-2-gazebo-unity/06-digital-twin',
        },
        {
          type: 'doc',
          id: 'module-2-gazebo-unity/07-gazebo-basics',
        },
        {
          type: 'doc',
          id: 'module-2-gazebo-unity/08-urdf-sdf-sim',
        },
        {
          type: 'doc',
          id: 'module-2-gazebo-unity/09-sensor-simulation',
        },
        {
          type: 'doc',
          id: 'module-2-gazebo-unity/10-unity-visualization',
        },
        {
          type: 'doc',
          id: 'module-2-gazebo-unity/module-2-summary',
        }
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)',
      items: [
        {
          type: 'doc',
          id: 'module-3-isaac/module-3-overview',
        },
        {
          type: 'doc',
          id: 'module-3-isaac/11-isaac-intro',
        },
        {
          type: 'doc',
          id: 'module-3-isaac/12-isaac-ros-perception',
        },
        {
          type: 'doc',
          id: 'module-3-isaac/13-nav2',
        },
        {
          type: 'doc',
          id: 'module-3-isaac/14-manipulation',
        },
        {
          type: 'doc',
          id: 'module-3-isaac/15-sim-to-real',
        }
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        {
          type: 'doc',
          id: 'module-4-vla/16-voice-to-action',
        },
        {
          type: 'doc',
          id: 'module-4-vla/17-llm-planning',
        },
        {
          type: 'doc',
          id: 'module-4-vla/18-vla-systems',
        },
        {
          type: 'doc',
          id: 'module-4-vla/19-conversational-robotics',
        },
        {
          type: 'doc',
          id: 'module-4-vla/20-capstone',
        }
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Appendices',
      items: [
        {
          type: 'doc',
          id: 'troubleshooting',
        },
        {
          type: 'doc',
          id: 'glossary',
        },
        {
          type: 'doc',
          id: 'references',
        }
      ],
      collapsed: true,
    }
  ],
};

module.exports = sidebars;
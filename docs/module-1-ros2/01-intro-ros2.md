---
title: "Introduction to ROS 2 Architecture"
description: "Understanding the fundamentals of ROS 2 for robotic systems"
slug: /module-1-ros2/01-intro-ros2
tags: [ROS2, Architecture, Robotics]
---

# Introduction to ROS 2 Architecture

This chapter introduces the Robot Operating System 2 (ROS 2), a flexible framework for writing robot software. ROS 2 is a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

## Learning Objectives

- Understand the basic concepts of ROS 2
- Learn about the architectural differences between ROS 1 and ROS 2
- Explore the DDS-based communication layer in ROS 2
- Understand Quality of Service (QoS) policies in ROS 2
- Set up a basic ROS 2 workspace and understand the development environment

## What is ROS 2?

ROS 2 is the next generation of the Robot Operating System, designed to be suitable for industrial use and commercial applications. It addresses many of the limitations of ROS 1, including improved security, real-time support, and better cross-platform compatibility. Unlike ROS 1 which relied on a centralized master architecture, ROS 2 uses a distributed architecture based on the Data Distribution Service (DDS) middleware.

ROS 2 represents a complete redesign of the original ROS framework with several key improvements:

- **Production readiness**: Built with industrial and commercial applications in mind
- **Security**: Built-in security features for safe deployment in production environments (Note: these must be explicitly configured and enabled)
- **Real-time support**: Enhanced capabilities for real-time systems
- **Cross-platform compatibility**: Improved support across different operating systems
- **Distributed architecture**: Elimination of the central master node dependency

## Architectural Evolution: From ROS 1 to ROS 2

### ROS 1 Architecture Limitations

ROS 1 utilized a peer-to-peer graph architecture with a central ROS Master that facilitated node registration and lookup. While effective for research and prototyping, this architecture had several limitations:

- **Single point of failure**: The ROS Master was critical to the system's operation
- **Limited scalability**: Difficult to scale across multiple machines reliably
- **No security**: No built-in authentication or authorization mechanisms
- **Poor real-time support**: Challenging to achieve deterministic real-time behavior
- **Network complexity**: Difficult to configure and maintain in complex network environments

### ROS 2 Architectural Improvements

ROS 2 addresses these limitations through a distributed architecture that leverages DDS (Data Distribution Service) as its underlying communication middleware. This design provides:

- **Decentralized operation**: No central master node required
- **Improved reliability**: Automatic discovery and fault tolerance
- **Enhanced security**: Built-in authentication, encryption, and access control (Note: these features must be explicitly configured and enabled)
- **Better real-time performance**: Deterministic communication patterns
- **Multi-vendor support**: Compatible with various DDS implementations

## Core Architecture Components

### DDS (Data Distribution Service)

At the heart of ROS 2 is the Data Distribution Service (DDS), an OMG (Object Management Group - an international standards organization that develops technology standards) standard for real-time, scalable, and fault-tolerant data exchange. DDS provides:

- **Automatic discovery**: Nodes automatically discover each other on the network
- **Reliable delivery**: Guaranteed message delivery with configurable policies
- **Quality of Service (QoS)**: Configurable behavior for different communication needs
- **Language independence**: Support for multiple programming languages
- **Platform neutrality**: Cross-platform compatibility

### RMW (ROS Middleware)

The ROS Middleware (RMW) layer acts as an abstraction between ROS 2 and the underlying DDS implementation. This allows ROS 2 to work with different DDS vendors while maintaining a consistent API. Popular RMW implementations include:

- **Fast DDS** (formerly Fast RTPS) - Default in many ROS 2 distributions
- **Cyclone DDS** - Lightweight and efficient
- **RTI Connext DDS** - Commercial solution with advanced features
- **OpenSplice DDS** - Open-source implementation

### Node Architecture

Nodes in ROS 2 are processes that perform computation and represent individual components of a robotic system. Each node:

- Encapsulates specific functionality (sensor processing, control algorithms, etc.)
- Communicates with other nodes through topics, services, or actions
- Maintains its own lifecycle and state
- Can be distributed across multiple machines

```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Minimal node created')

def main(args=None):
    rclpy.init(args=args)
    node = MinimalNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Quality of Service (QoS) Policies

One of the key architectural innovations in ROS 2 is the Quality of Service (QoS) system, which allows fine-grained control over communication behavior:

### Reliability Policy
- **Reliable**: All messages are guaranteed to be delivered
- **Best Effort**: Messages may be dropped without notification

### Durability Policy
- **Transient Local**: Historical data is available to late-joining subscribers
- **Volatile**: Only new data is sent to subscribers

### Deadline Policy
Defines the maximum time interval between consecutive messages.

### Lifespan Policy
Specifies how long a message remains valid after publication.

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Configure QoS for sensor data (may drop messages but low latency)
sensor_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE
)

# Configure QoS for critical commands (must arrive reliably)
command_qos = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL
)
```

## ROS 2 Ecosystem and Tools

### Command Line Tools

ROS 2 provides a comprehensive set of command-line tools for development and debugging:

- `ros2 run`: Execute a node
- `ros2 topic`: Inspect and interact with topics
- `ros2 service`: Work with services
- `ros2 action`: Manage actions
- `ros2 node`: Monitor and control nodes
- `ros2 param`: Handle parameters
- `ros2 pkg`: Package management

### Development Environment

Setting up a ROS 2 development environment involves:

1. **Installation**: Choose a ROS 2 distribution (Humble Hawksbill, Iron Irwin, Jazzy Jalisco, etc.)
2. **Workspace creation**: Organize packages in a colcon workspace
3. **Environment setup**: Source the ROS 2 installation

```bash
# Create a workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build the workspace
colcon build

# Source the environment
source install/setup.bash
```

## Practical Exercise: Setting Up Your First ROS 2 Workspace

Let's create a simple ROS 2 package to demonstrate the architecture:

```bash
# Navigate to your workspace
cd ~/ros2_ws/src

# Create a new package
ros2 pkg create --build-type ament_python my_robot_package

# Change to the package directory
cd my_robot_package
```

Create a simple publisher node (`my_robot_package/my_robot_package/simple_publisher.py`):

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SimplePublisher(Node):
    def __init__(self):
        super().__init__('simple_publisher')
        self.publisher = self.create_publisher(String, 'robot_status', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Robot status: operational {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    simple_publisher = SimplePublisher()

    try:
        rclpy.spin(simple_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        simple_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Update the package's `setup.py` to include the executable:

```python
entry_points={
    'console_scripts': [
        'simple_publisher = my_robot_package.simple_publisher:main',
    ],
},
```

Build and run the package:

```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_package
source install/setup.bash
ros2 run my_robot_package simple_publisher
```

In another terminal, listen to the published messages:

```bash
source ~/ros2_ws/install/setup.bash
ros2 topic echo /robot_status std_msgs/msg/String
```

## Troubleshooting Common Issues

### Common Errors and Solutions

1. **Package not found error**:
   - Make sure you've sourced the workspace: `source install/setup.bash`
   - Ensure the package was built successfully: `colcon build`

2. **Node not connecting to ROS graph**:
   - Check that the ROS_DOMAIN_ID environment variable is consistent across all terminals
   - Ensure no firewall is blocking ROS communication

3. **Permission denied when building**:
   - Make sure you have write permissions to the workspace directory
   - Avoid using sudo with ROS commands

## Key Differences from ROS 1

| Feature | ROS 1 | ROS 2 |
|---------|-------|-------|
| Architecture | Centralized (master) | Distributed (DDS) |
| Communication | TCPROS/UDPROS | DDS-based |
| Security | None | Built-in security |
| Real-time | Limited | Enhanced support |
| Multi-machine | Complex setup | Automatic discovery |
| Language support | Limited | Extensive |

## Summary

ROS 2 represents a significant architectural evolution from ROS 1, focusing on production readiness, security, and scalability. The DDS-based communication layer provides a solid foundation for distributed robotic systems, while the QoS system allows for fine-grained control over communication behavior. Understanding these architectural concepts is essential for developing robust robotic applications with ROS 2.

The distributed architecture eliminates single points of failure, making ROS 2 suitable for industrial and commercial applications. The modular design allows for flexibility in choosing middleware implementations while maintaining API consistency through the RMW layer.

## Key Takeaways

- ROS 2 uses a distributed architecture based on DDS middleware
- Quality of Service (QoS) policies provide configurable communication behavior
- Security features are built into the core architecture but require explicit configuration
- Multiple DDS implementations are supported through the RMW abstraction
- The development workflow emphasizes colcon-based workspace management
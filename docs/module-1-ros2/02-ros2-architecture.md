---
title: "ROS 2 Communication Patterns: Nodes, Topics, Services, and Actions"
description: "Deep dive into ROS 2 communication patterns and their practical applications"
slug: /module-1-ros2/02-ros2-architecture
tags: [ROS2, Nodes, Topics, Services, Actions, Communication]
---

# ROS 2 Communication Patterns: Nodes, Topics, Services, and Actions

This chapter explores the core communication patterns in ROS 2: nodes, topics, services, and actions. Understanding these concepts is crucial for building distributed robotic systems that can effectively coordinate and share information.

## Learning Objectives

- Understand the role of nodes in ROS 2
- Learn about topic-based communication (publish/subscribe)
- Explore service-based communication (request/response)
- Understand actions for long-running tasks with feedback
- Implement communication patterns in practical examples
- Choose appropriate communication patterns for different use cases

## Nodes: The Fundamental Building Blocks

A node is a process that performs computation. Nodes are the fundamental building blocks of a ROS 2 system. Each node can perform specific tasks and communicate with other nodes through topics, services, or actions.

### Node Lifecycle

ROS 2 nodes have a well-defined lifecycle that includes several states:

- **Unconfigured**: Node created but not yet configured
- **Inactive**: Node configured but not active
- **Active**: Node running and performing its function
- **Finalized**: Node has been shut down and cleaned up

```python
import rclpy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn

class LifecycleMinimalNode(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_minimal_node')

    def on_configure(self, state):
        self.get_logger().info('Configuring node')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info('Activating node')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.get_logger().info('Deactivating node')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self.get_logger().info('Cleaning up node')
        return TransitionCallbackReturn.SUCCESS
```

### Node Parameters

Nodes can be configured using parameters that can be set at runtime:

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'my_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_mode', True)

        # Access parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.safety_mode = self.get_parameter('safety_mode').value

        self.get_logger().info(f'Robot: {self.robot_name}, Max velocity: {self.max_velocity}')
```

## Topic-Based Communication (Publish/Subscribe)

Topics enable asynchronous, many-to-many communication through a publish/subscribe pattern. Publishers send messages to topics, and subscribers receive messages from topics without direct knowledge of each other.

### Publisher Implementation

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')

        # Create publishers for different sensor data
        self.temperature_pub = self.create_publisher(Float64, 'temperature', 10)
        self.status_pub = self.create_publisher(String, 'robot_status', 10)

        # Create a timer to publish data periodically
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        # Publish temperature data
        temp_msg = Float64()
        temp_msg.data = 25.0 + (self.i * 0.5)  # Simulated temperature
        self.temperature_pub.publish(temp_msg)

        # Publish status message
        status_msg = String()
        status_msg.data = f'Robot operational - cycle {self.i}'
        self.status_pub.publish(status_msg)

        self.get_logger().info(f'Published: temp={temp_msg.data}°C, status={status_msg.data}')
        self.i += 1
```

### Subscriber Implementation

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64

class DataSubscriber(Node):
    def __init__(self):
        super().__init__('data_subscriber')

        # Create subscribers for different topics
        self.temperature_sub = self.create_subscription(
            Float64,
            'temperature',
            self.temperature_callback,
            10)

        self.status_sub = self.create_subscription(
            String,
            'robot_status',
            self.status_callback,
            10)

    def temperature_callback(self, msg):
        self.get_logger().info(f'Received temperature: {msg.data}°C')

        # Process temperature data (e.g., check for overheating)
        if msg.data > 80.0:
            self.get_logger().warn('Temperature exceeds safe limit!')

    def status_callback(self, msg):
        self.get_logger().info(f'Received status: {msg.data}')
```

### Topic Commands and Tools

Useful command-line tools for working with topics:

```bash
# List all topics
ros2 topic list

# Show topic information
ros2 topic info /temperature

# Echo topic messages
ros2 topic echo /robot_status std_msgs/msg/String

# Publish a message to a topic
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 1.0}, angular: {z: 0.5}}'

# Show topic statistics
ros2 topic hz /temperature
```

## Service-Based Communication (Request/Response)

Services provide synchronous, request/response communication between nodes. A client sends a request to a service and waits for a response.

### Service Definition

First, create a service definition file (`srv/AddTwoInts.srv`):

```
int64 a
int64 b
---
int64 sum
```

### Service Implementation

```python
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {request.a} + {request.b} = {response.sum}')
        return response
```

### Service Client Implementation

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

### Service Commands and Tools

```bash
# List all services
ros2 service list

# Show service information
ros2 service info /add_two_ints

# Call a service
ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts '{a: 1, b: 2}'
```

## Action-Based Communication

Actions are designed for long-running tasks that require feedback and the ability to cancel. They combine the benefits of topics and services.

### Action Definition

Create an action definition file (`action/Fibonacci.action`):

```
int32 order
---
int32[] sequence
---
int32[] partial_sequence
```

### Action Server Implementation

```python
import time
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            callback_group=rclpy.callback_groups.ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.partial_sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.partial_sequence.append(
                feedback_msg.partial_sequence[i] + feedback_msg.partial_sequence[i-1])

            self.get_logger().info(f'Publishing feedback: {feedback_msg.partial_sequence}')
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.partial_sequence
        self.get_logger().info(f'Returning result: {result.sequence}')

        return result
```

### Action Client Implementation

```python
import time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(self, Fibonacci, 'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Received feedback: {feedback_msg.feedback.partial_sequence}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')
```

## Choosing the Right Communication Pattern

### When to Use Each Pattern

| Pattern | Use Case | Characteristics |
|---------|----------|-----------------|
| **Topics** | Sensor data, status updates, continuous streams | Asynchronous, many-to-many, fire-and-forget |
| **Services** | Query-response, configuration, simple commands | Synchronous, one-to-one, request-response |
| **Actions** | Long-running tasks, tasks with feedback/cancel | Asynchronous, one-to-one, with feedback and cancelation |

### Practical Decision Framework

1. **Use Topics when:**
   - Broadcasting sensor data or status
   - Multiple subscribers need the same information
   - Real-time streaming is required
   - No acknowledgment needed

2. **Use Services when:**
   - Requesting specific computation
   - Need immediate response
   - Simple command execution
   - Synchronous operation required

3. **Use Actions when:**
   - Task takes significant time to complete
   - Progress feedback is needed
   - Task may need to be canceled
   - Complex state management required

## Practical Exercise: Implementing a Robot Control System

Let's create a complete example that demonstrates all three communication patterns working together in a robot control system:

### Robot Status Publisher (Topic)

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import random

class RobotStatusPublisher(Node):
    def __init__(self):
        super().__init__('robot_status_publisher')
        self.publisher = self.create_publisher(String, 'robot_status', 10)
        self.velocity_sub = self.create_subscription(
            Twist, 'cmd_vel', self.velocity_callback, 10)

        timer_period = 2.0
        self.timer = self.create_timer(timer_period, self.status_callback)
        self.current_status = "IDLE"

    def velocity_callback(self, msg):
        if msg.linear.x != 0.0 or msg.angular.z != 0.0:
            self.current_status = "MOVING"
        else:
            self.current_status = "STOPPED"

    def status_callback(self):
        msg = String()
        msg.data = f'Robot status: {self.current_status}, battery: {random.randint(20, 100)}%'
        self.publisher.publish(msg)
        self.get_logger().info(f'Published status: {msg.data}')
```

### Robot Control Service

```python
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist
from example_interfaces.srv import Trigger

class RobotControlService(Node):
    def __init__(self):
        super().__init__('robot_control_service')
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.srv = self.create_service(Trigger, 'robot_stop', self.stop_callback)

    def stop_callback(self, request, response):
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(stop_msg)

        response.success = True
        response.message = "Robot stopped successfully"

        self.get_logger().info('Stopping robot')
        return response
```

### Navigation Action Server

```python
import time
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from geometry_msgs.msg import Twist
from example_interfaces.action import NavigateToPose

class NavigationActionServer(Node):
    def __init__(self):
        super().__init__('navigation_action_server')
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            execute_callback=self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info(f'Navigating to pose: {goal_handle.request.pose}')

        # Simulate navigation by moving forward
        move_cmd = Twist()
        move_cmd.linear.x = 0.5  # Move forward at 0.5 m/s
        move_cmd.angular.z = 0.0

        # Simulate navigation for a period of time
        for i in range(10):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Navigation canceled')
                return NavigateToPose.Result()

            self.cmd_vel_pub.publish(move_cmd)
            time.sleep(0.5)

        # Stop the robot
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(stop_cmd)

        goal_handle.succeed()
        result = NavigateToPose.Result()
        result.success = True
        self.get_logger().info('Navigation completed successfully')

        return result
```

## Quality of Service Considerations

When implementing communication patterns, consider the appropriate QoS settings:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# For sensor data (real-time, may drop messages)
sensor_qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE
)

# For critical commands (must be reliable)
command_qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE
)

# For configuration parameters (must persist)
config_qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL
)
```

## Summary

ROS 2 provides three distinct communication patterns, each suited for different use cases:

- **Topics** provide asynchronous, publish-subscribe communication ideal for sensor data and status updates
- **Services** offer synchronous request-response communication for immediate queries and commands
- **Actions** enable long-running operations with feedback and cancelation capabilities

Understanding when to use each pattern is crucial for designing effective robotic systems. The choice of communication pattern affects system performance, reliability, and maintainability.

## Key Takeaways

- Topics are best for continuous data streams and broadcasting
- Services are ideal for immediate request-response interactions
- Actions are designed for complex, long-running tasks
- Quality of Service settings allow fine-tuning communication behavior
- Proper selection of communication patterns is essential for robust system design
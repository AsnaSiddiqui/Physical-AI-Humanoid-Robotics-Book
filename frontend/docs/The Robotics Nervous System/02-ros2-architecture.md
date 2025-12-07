---
sidebar_position: 6
title: "ROS 2 Communication Patterns: Nodes, Topics, Services, and Actions"
description: "Deep dive into ROS 2 communication patterns for humanoid robotics applications"
keywords: [ROS2, Nodes, Topics, Services, Actions, Communication, Humanoid Robotics]
---

# ROS 2 Communication Patterns: Nodes, Topics, Services, and Actions

This chapter explores the core communication patterns in ROS 2: nodes, topics, services, and actions. Understanding these concepts is crucial for building distributed humanoid robotic systems that can effectively coordinate and share information.

## Learning Objectives

- Understand the role of nodes in ROS 2
- Learn about topic-based communication (publish/subscribe)
- Explore service-based communication (request/response)
- Understand actions for long-running tasks with feedback
- Implement communication patterns in practical humanoid robotics examples
- Choose appropriate communication patterns for different use cases

## Nodes: The Fundamental Building Blocks

A node is a process that performs computation. Nodes are the fundamental building blocks of a ROS 2 system. Each node can perform specific tasks and communicate with other nodes through topics, services, or actions.

### Node Lifecycle

ROS 2 nodes have a well-defined lifecycle that includes several states:

- **Unconfigured**: Node created but not yet configured
- **Inactive**: Node configured but not active
- **Active**: Node running and performing its function
- **Finalized**: Node has been shut down and cleaned up

For humanoid robotics, the lifecycle concept is especially important as it allows for graceful transitions between different operational states (e.g., from standby to active movement).

```python
import rclpy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.qos import QoSProfile

class HumanoidLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('humanoid_lifecycle_node')
        self.get_logger().info('Lifecycle node created')

    def on_configure(self, state):
        self.get_logger().info('Configuring humanoid node')

        # Create publishers and subscribers
        self.joint_state_pub = self.create_publisher(
            JointState,
            'joint_states',
            QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        )

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info('Activating humanoid node')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.get_logger().info('Deactivating humanoid node')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self.get_logger().info('Cleaning up humanoid node')
        return TransitionCallbackReturn.SUCCESS
```

### Node Parameters

Nodes can be configured using parameters that can be set at runtime, which is particularly useful for humanoid robots that may need to adjust their behavior based on different environments or tasks:

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class HumanoidParameterNode(Node):
    def __init__(self):
        super().__init__('humanoid_parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'humanoid_bot')
        self.declare_parameter('walking_speed', 0.5)  # m/s
        self.declare_parameter('safety_mode', True)
        self.declare_parameter('torso_height', 0.8)  # meters

        # Access parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.walking_speed = self.get_parameter('walking_speed').value
        self.safety_mode = self.get_parameter('safety_mode').value
        self.torso_height = self.get_parameter('torso_height').value

        self.get_logger().info(f'Robot: {self.robot_name}, Speed: {self.walking_speed}m/s')

        # Create a parameter callback to handle dynamic changes
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        """Handle parameter changes during runtime"""
        for param in params:
            if param.name == 'walking_speed':
                if param.value > 2.0:  # Safety check
                    return SetParametersResult(successful=False, reason='Speed too high')
                else:
                    self.walking_speed = param.value
                    self.get_logger().info(f'Walking speed updated to {param.value}m/s')

        return SetParametersResult(successful=True)
```

## Topic-Based Communication (Publish/Subscribe)

Topics enable asynchronous, many-to-many communication through a publish/subscribe pattern. This is the primary communication method for sensor data and state information in humanoid robotics.

### Publisher Implementation

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Vector3Stamped
from builtin_interfaces.msg import Time
import math

class HumanoidSensorPublisher(Node):
    def __init__(self):
        super().__init__('humanoid_sensor_publisher')

        # Create publishers for different sensor data
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu/data', 10)
        self.center_of_mass_pub = self.create_publisher(Vector3Stamped, 'center_of_mass', 10)

        # Create a timer to publish data periodically (50Hz for humanoid control)
        timer_period = 0.02  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Initialize joint names for a humanoid robot
        self.joint_names = [
            'left_hip', 'left_knee', 'left_ankle',
            'right_hip', 'right_knee', 'right_ankle',
            'left_shoulder', 'left_elbow', 'left_wrist',
            'right_shoulder', 'right_elbow', 'right_wrist',
            'torso_yaw', 'torso_pitch', 'torso_roll'
        ]

        self.joint_positions = [0.0] * len(self.joint_names)
        self.time_step = 0

    def timer_callback(self):
        # Publish joint state data
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.header.frame_id = 'base_link'
        joint_msg.name = self.joint_names

        # Simulate changing joint positions (e.g., walking pattern)
        for i in range(len(self.joint_positions)):
            self.joint_positions[i] = math.sin(self.time_step * 0.1 + i * 0.5) * 0.2

        joint_msg.position = self.joint_positions
        self.joint_state_pub.publish(joint_msg)

        # Publish IMU data
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'imu_link'

        # Simulate IMU readings
        imu_msg.linear_acceleration.x = math.sin(self.time_step * 0.05) * 0.1
        imu_msg.linear_acceleration.y = math.cos(self.time_step * 0.05) * 0.1
        imu_msg.linear_acceleration.z = 9.81 + math.sin(self.time_step * 0.1) * 0.05

        self.imu_pub.publish(imu_msg)

        self.time_step += 1

        self.get_logger().info(f'Published sensor data - Joint positions: {self.joint_positions[:3]}...')
```

### Subscriber Implementation

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import numpy as np

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Create subscribers for different topics
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            'imu/data',
            self.imu_callback,
            10
        )

        self.command_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.command_callback,
            10
        )

        # Publisher for motor commands
        self.motor_command_pub = self.create_publisher(JointState, 'motor_commands', 10)

        # Internal state
        self.current_joints = {}
        self.balance_state = {'stable': True, 'tilt_angle': 0.0}

    def joint_state_callback(self, msg):
        """Process incoming joint state data"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joints[name] = msg.position[i]

        # Check balance based on joint positions
        self.check_balance()

    def imu_callback(self, msg):
        """Process IMU data for balance control"""
        # Calculate tilt angle from IMU data
        tilt_x = math.atan2(msg.linear_acceleration.y, msg.linear_acceleration.z)
        tilt_y = math.atan2(-msg.linear_acceleration.x,
                            math.sqrt(msg.linear_acceleration.y**2 + msg.linear_acceleration.z**2))

        self.balance_state['tilt_angle'] = math.sqrt(tilt_x**2 + tilt_y**2)
        self.balance_state['stable'] = self.balance_state['tilt_angle'] < 0.1  # 0.1 rad threshold

        if not self.balance_state['stable']:
            self.get_logger().warning(f'Balance compromised! Tilt: {self.balance_state["tilt_angle"]:.3f}')

    def command_callback(self, msg):
        """Process velocity commands"""
        if self.balance_state['stable']:
            # Generate motor commands based on velocity command
            self.generate_motor_commands(msg.linear.x, msg.angular.z)
        else:
            # Emergency stabilization
            self.emergency_stabilization()

    def check_balance(self):
        """Check robot balance based on joint positions"""
        # Simplified balance check - in real robots this would be more complex
        left_foot_pos = self.current_joints.get('left_ankle', 0.0)
        right_foot_pos = self.current_joints.get('right_ankle', 0.0)

        # Check if feet are properly positioned for stability
        if abs(left_foot_pos - right_foot_pos) > 0.5:  # 50cm threshold
            self.get_logger().warning('Foot positioning may affect stability')

    def generate_motor_commands(self, linear_vel, angular_vel):
        """Generate motor commands based on desired motion"""
        cmd_msg = JointState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()

        # Simplified gait generation
        # In practice, this would use inverse kinematics and gait planners
        cmd_msg.name = list(self.current_joints.keys())
        cmd_msg.position = []  # Would contain target positions

        for joint_name in cmd_msg.name:
            # Calculate target position based on gait and balance
            target_pos = self.calculate_target_position(joint_name, linear_vel, angular_vel)
            cmd_msg.position.append(target_pos)

        self.motor_command_pub.publish(cmd_msg)

    def emergency_stabilization(self):
        """Emergency stabilization routine"""
        self.get_logger().info('Initiating emergency stabilization')
        # Send commands to return to stable stance
        # Implementation would depend on specific robot hardware
```

### Topic Commands and Tools

Useful command-line tools for working with topics in humanoid robotics:

```bash
# List all topics
ros2 topic list

# Show topic information and message type
ros2 topic info /joint_states

# Echo topic messages (useful for debugging)
ros2 topic echo /joint_states sensor_msgs/msg/JointState

# Echo with frequency information
ros2 topic hz /imu/data

# Echo with bandwidth information
ros2 topic bw /camera/image_raw

# Publish a message to a topic (for testing)
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.2}}'
```

## Service-Based Communication (Request/Response)

Services provide synchronous, request/response communication between nodes. This is ideal for humanoid robotics tasks that require immediate responses.

### Service Definition

First, create a service definition file (`srv/SetJointStiffness.srv`):

```
# Joint stiffness values for compliance control
string[] joint_names
float64[] stiffness_values
---
bool success
string message
```

### Service Implementation

```python
from rclpy.node import Node
from rclpy.qos import QoSProfile
from humanoid_interfaces.srv import SetJointStiffness  # Custom service
import threading

class HumanoidComplianceController(Node):
    def __init__(self):
        super().__init__('humanoid_compliance_controller')
        self.srv = self.create_service(
            SetJointStiffness,
            'set_joint_stiffness',
            self.set_joint_stiffness_callback
        )

        # Dictionary to store current joint stiffness values
        self.joint_stiffness = {}

    def set_joint_stiffness_callback(self, request, response):
        """Set joint stiffness for compliant control"""
        try:
            if len(request.joint_names) != len(request.stiffness_values):
                response.success = False
                response.message = 'Mismatch between joint names and stiffness values count'
                return response

            # Update stiffness values
            for joint_name, stiffness in zip(request.joint_names, request.stiffness_values):
                # Validate stiffness range (0.0 to 100.0 for example)
                if 0.0 <= stiffness <= 100.0:
                    self.joint_stiffness[joint_name] = stiffness
                    self.get_logger().info(f'Set {joint_name} stiffness to {stiffness}')
                else:
                    response.success = False
                    response.message = f'Stiffness value {stiffness} for {joint_name} is out of range [0, 100]'
                    return response

            response.success = True
            response.message = f'Successfully updated stiffness for {len(request.joint_names)} joints'

            # Apply the new stiffness values to the hardware
            self.apply_stiffness_values()

        except Exception as e:
            response.success = False
            response.message = f'Exception occurred: {str(e)}'

        return response

    def apply_stiffness_values(self):
        """Apply stiffness values to the actual robot hardware"""
        # Implementation would interface with robot's control system
        # This is where the actual stiffness values would be sent to motors
        pass
```

### Service Client Implementation

```python
import rclpy
from rclpy.node import Node
from humanoid_interfaces.srv import SetJointStiffness
import time

class StiffnessClient(Node):
    def __init__(self):
        super().__init__('stiffness_client')
        self.cli = self.create_client(SetJointStiffness, 'set_joint_stiffness')

        # Wait for the service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for set_joint_stiffness service...')

        self.req = SetJointStiffness.Request()

    def send_request(self, joint_names, stiffness_values):
        """Send stiffness update request"""
        self.req.joint_names = joint_names
        self.req.stiffness_values = stiffness_values

        self.future = self.cli.call_async(self.req)
        return self.future

def main(args=None):
    rclpy.init(args=args)
    stiffness_client = StiffnessClient()

    # Example: Set different stiffness for different joints
    joint_names = ['left_leg', 'right_leg', 'left_arm', 'right_arm']
    stiffness_values = [80.0, 80.0, 40.0, 40.0]  # Higher for legs, lower for arms

    future = stiffness_client.send_request(joint_names, stiffness_values)

    try:
        rclpy.spin_until_future_complete(stiffness_client, future)
        response = future.result()

        if response.success:
            print(f'Success: {response.message}')
        else:
            print(f'Failed: {response.message}')

    except Exception as e:
        print(f'Service call failed: {e}')
    finally:
        stiffness_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Commands and Tools

```bash
# List all services
ros2 service list

# Show service information
ros2 service info /set_joint_stiffness

# Call a service directly from command line
ros2 service call /set_joint_stiffness humanoid_interfaces/srv/SetJointStiffness '{joint_names: ["left_leg", "right_leg"], stiffness_values: [80.0, 80.0]}'
```

## Action-Based Communication

Actions are designed for long-running tasks that require feedback and the ability to cancel. This is perfect for humanoid robotics tasks like walking, manipulation, or navigation.

### Action Definition

Create an action definition file (`action/WalkToPose.action`):

```
# Goal: Target pose for the humanoid to walk to
geometry_msgs/Pose target_pose
float32 step_size
---
# Result: Whether the walk was successful
bool success
float32 distance_traveled
string final_status
---
# Feedback: Current progress of the walk
float32 distance_remaining
float32 progress_percentage
geometry_msgs/Pose current_pose
```

### Action Server Implementation

```python
import time
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from geometry_msgs.msg import Pose
from humanoid_interfaces.action import WalkToPose
import math

class HumanoidWalkActionServer(Node):
    def __init__(self):
        super().__init__('humanoid_walk_action_server')
        self._action_server = ActionServer(
            self,
            WalkToPose,
            'walk_to_pose',
            execute_callback=self.execute_callback,
            callback_group=rclpy.callback_groups.ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Publisher for actual robot movement
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Current pose tracker
        self.current_pose = Pose()
        self.is_moving = False

    def goal_callback(self, goal_request):
        """Accept or reject goal requests"""
        # Check if the target is reachable
        distance = self.calculate_distance(self.current_pose, goal_request.target_pose)

        if distance > 10.0:  # 10 meter limit for safety
            self.get_logger().info('Goal rejected: too far away')
            return GoalResponse.REJECT
        else:
            self.get_logger().info('Goal accepted')
            return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle goal cancellation requests"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def calculate_distance(self, pose1, pose2):
        """Calculate Euclidean distance between two poses"""
        dx = pose2.position.x - pose1.position.x
        dy = pose2.position.y - pose1.position.y
        dz = pose2.position.z - pose1.position.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    async def execute_callback(self, goal_handle):
        """Execute the walking goal"""
        self.get_logger().info('Starting walk to pose...')

        # Get the goal
        target_pose = goal_handle.request.target_pose
        step_size = goal_handle.request.step_size

        # Calculate total distance
        total_distance = self.calculate_distance(self.current_pose, target_pose)

        # Initialize feedback
        feedback_msg = WalkToPose.Feedback()
        feedback_msg.distance_remaining = total_distance
        feedback_msg.progress_percentage = 0.0
        feedback_msg.current_pose = self.current_pose

        # Simulate walking (in real robot, this would interface with gait planner)
        step_count = 0
        max_steps = int(total_distance / step_size) if step_size > 0 else 0

        for step in range(max_steps):
            # Check if goal was cancelled
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled during execution')

                # Stop the robot
                self.publish_stop_command()

                result = WalkToPose.Result()
                result.success = False
                result.distance_traveled = step * step_size
                result.final_status = 'CANCELED'

                return result

            # Simulate movement - update current pose
            ratio = step / max_steps
            self.current_pose.position.x = (
                self.current_pose.position.x * (1 - ratio) +
                target_pose.position.x * ratio
            )
            self.current_pose.position.y = (
                self.current_pose.position.y * (1 - ratio) +
                target_pose.position.y * ratio
            )

            # Calculate remaining distance
            remaining_distance = self.calculate_distance(self.current_pose, target_pose)

            # Update feedback
            feedback_msg.distance_remaining = remaining_distance
            feedback_msg.progress_percentage = (step / max_steps) * 100.0
            feedback_msg.current_pose = self.current_pose

            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)

            # Publish velocity command to move the robot
            self.publish_movement_command(target_pose, step_size)

            # Sleep to simulate real walking time
            time.sleep(0.1)

        # Goal succeeded
        goal_handle.succeed()

        result = WalkToPose.Result()
        result.success = True
        result.distance_traveled = total_distance
        result.final_status = 'REACHED_GOAL'

        self.get_logger().info(f'Walk completed successfully. Traveled: {result.distance_traveled:.2f}m')

        # Stop the robot after reaching the goal
        self.publish_stop_command()

        return result

    def publish_movement_command(self, target_pose, step_size):
        """Publish velocity commands to move the robot"""
        cmd_msg = Twist()

        # Calculate direction to target
        dx = target_pose.position.x - self.current_pose.position.x
        dy = target_pose.position.y - self.current_pose.position.y
        distance = math.sqrt(dx*dx + dy*dy)

        if distance > 0.1:  # If not close enough to target
            # Normalize direction and scale by step size
            cmd_msg.linear.x = min(step_size, 0.5) * (dx / distance)  # Limit max speed
            cmd_msg.linear.y = min(step_size, 0.5) * (dy / distance)

            # Add some rotation to face the target
            target_yaw = math.atan2(dy, dx)
            # Simplified rotation command - real robot would use orientation
            cmd_msg.angular.z = target_yaw * 0.1  # Rotation gain

        self.cmd_vel_pub.publish(cmd_msg)

    def publish_stop_command(self):
        """Publish zero velocity to stop the robot"""
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0
        cmd_msg.linear.y = 0.0
        cmd_msg.linear.z = 0.0
        cmd_msg.angular.x = 0.0
        cmd_msg.angular.y = 0.0
        cmd_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_msg)
```

### Action Client Implementation

```python
import time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Pose
from humanoid_interfaces.action import WalkToPose

class HumanoidWalkActionClient(Node):
    def __init__(self):
        super().__init__('humanoid_walk_action_client')
        self._action_client = ActionClient(self, WalkToPose, 'walk_to_pose')

    def send_goal(self, target_x, target_y, step_size=0.1):
        """Send a goal to walk to a specific position"""
        goal_msg = WalkToPose.Goal()

        # Set target pose
        goal_msg.target_pose = Pose()
        goal_msg.target_pose.position.x = target_x
        goal_msg.target_pose.position.y = target_y
        goal_msg.target_pose.position.z = 0.0
        # Set orientation (facing forward)
        goal_msg.target_pose.orientation.w = 1.0

        goal_msg.step_size = step_size

        self.get_logger().info(f'Sending goal to walk to ({target_x}, {target_y})')

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """Handle feedback during action execution"""
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'Feedback: {feedback.progress_percentage:.1f}% complete, ' +
            f'{feedback.distance_remaining:.2f}m remaining'
        )

    def get_result_callback(self, future):
        """Handle action result"""
        result = future.result().result
        self.get_logger().info(f'Result: Success={result.success}, Distance={result.distance_traveled:.2f}m')

        # Shutdown after completing the action
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)

    action_client = HumanoidWalkActionClient()

    # Send goal to walk to position (2.0, 1.0) with 0.1m steps
    action_client.send_goal(2.0, 1.0, 0.1)

    # Spin to process callbacks
    rclpy.spin(action_client)

if __name__ == '__main__':
    main()
```

## Choosing the Right Communication Pattern

### When to Use Each Pattern

| Pattern | Use Case | Characteristics |
|---------|----------|-----------------|
| **Topics** | Sensor data, status updates, continuous streams | Asynchronous, many-to-many, fire-and-forget |
| **Services** | Query-response, configuration, simple commands | Synchronous, one-to-one, request-response |
| **Actions** | Long-running tasks, tasks with feedback/cancelation | Asynchronous, one-to-one, with feedback and cancelation |

### Practical Decision Framework

1. **Use Topics when:**
   - Broadcasting sensor data or status (IMU, joint states, camera feeds)
   - Multiple subscribers need the same information
   - Real-time streaming is required
   - No acknowledgment needed
   - Example: Publishing joint states from sensors, broadcasting robot status

2. **Use Services when:**
   - Requesting specific computation (inverse kinematics)
   - Need immediate response (calibration, configuration)
   - Simple command execution (set joint stiffness, save calibration)
   - Synchronous operation required
   - Example: Calibrating sensors, configuring robot parameters

3. **Use Actions when:**
   - Task takes significant time to complete (walking to location)
   - Progress feedback is needed (navigation progress)
   - Task may need to be canceled (emergency stop during navigation)
   - Complex state management required (manipulation with multiple steps)
   - Example: Walking to a pose, grasping an object, executing a complex behavior

## Practical Exercise: Implementing a Humanoid Balance Controller

Let's create a complete example that demonstrates all three communication patterns working together in a humanoid robot balance control system:

### Balance Controller Node

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32
from humanoid_interfaces.srv import SetJointStiffness
from humanoid_interfaces.action import EmergencyStance
from rclpy.action import ActionServer
import numpy as np
import math

class HumanoidBalanceController(Node):
    def __init__(self):
        super().__init__('humanoid_balance_controller')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.joint_cmd_pub = self.create_publisher(JointState, 'joint_commands', 10)
        self.balance_status_pub = self.create_publisher(Bool, 'balance_stable', 10)

        # Subscribers
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)

        # Service server
        self.stiffness_srv = self.create_service(
            SetJointStiffness, 'adjust_balance_stiffness', self.adjust_stiffness_callback)

        # Action server for emergency stance
        self.emergency_action_server = ActionServer(
            self, EmergencyStance, 'emergency_stance', self.emergency_stance_callback)

        # Internal state
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.joint_positions = {}
        self.balance_threshold = 0.15  # radians

        # Balance control timer (100Hz for fast response)
        self.balance_timer = self.create_timer(0.01, self.balance_control_loop)

        self.get_logger().info('Humanoid Balance Controller initialized')

    def imu_callback(self, msg):
        """Process IMU data for balance estimation"""
        # Convert quaternion to Euler angles (simplified)
        w, x, y, z = msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z

        # Roll (rotation around X-axis)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        self.roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (rotation around Y-axis)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            self.pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            self.pitch = math.asin(sinp)

        # Yaw (rotation around Z-axis)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def joint_state_callback(self, msg):
        """Update joint position dictionary"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]

    def balance_control_loop(self):
        """Main balance control loop"""
        # Calculate balance error
        balance_error = math.sqrt(self.roll**2 + self.pitch**2)

        # Check if we're out of balance
        is_unstable = balance_error > self.balance_threshold

        # Publish balance status
        status_msg = Bool()
        status_msg.data = not is_unstable
        self.balance_status_pub.publish(status_msg)

        if is_unstable:
            self.get_logger().warning(f'Balance compromised! Angle: {math.degrees(balance_error):.2f}°')
            self.correct_balance()
        else:
            # Normal walking behavior when balanced
            self.normal_locomotion()

    def correct_balance(self):
        """Correct robot balance by adjusting joint positions"""
        cmd_msg = JointState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()

        # Calculate corrective movements based on roll/pitch
        roll_correction = -self.roll * 10.0  # Gain for roll correction
        pitch_correction = -self.pitch * 10.0  # Gain for pitch correction

        # Adjust ankle joints for balance correction
        cmd_msg.name = ['left_ankle_roll', 'left_ankle_pitch', 'right_ankle_roll', 'right_ankle_pitch']
        cmd_msg.position = [
            roll_correction * 0.5,   # Left ankle roll
            pitch_correction * 0.5,  # Left ankle pitch
            -roll_correction * 0.5,  # Right ankle roll (opposite)
            pitch_correction * 0.5   # Right ankle pitch
        ]

        # Also adjust hip joints slightly
        cmd_msg.name.extend(['left_hip_roll', 'right_hip_roll'])
        cmd_msg.position.extend([
            roll_correction * 0.3,   # Left hip roll
            -roll_correction * 0.3   # Right hip roll (opposite)
        ])

        self.joint_cmd_pub.publish(cmd_msg)

    def normal_locomotion(self):
        """Normal walking/locomotion behavior when balanced"""
        # For now, just maintain current position
        # In a real implementation, this would handle normal walking gaits
        pass

    def adjust_stiffness_callback(self, request, response):
        """Service callback to adjust joint stiffness for balance"""
        # In a real robot, this would interface with the motor control system
        # to adjust PID gains or stiffness parameters

        try:
            # Increase stiffness when balance is challenged
            for joint_name in request.joint_names:
                if joint_name in ['left_ankle', 'right_ankle', 'left_hip', 'right_hip']:
                    # These joints are critical for balance
                    self.get_logger().info(f'Increasing stiffness for {joint_name}')

            response.success = True
            response.message = f'Adjusted stiffness for {len(request.joint_names)} joints'
        except Exception as e:
            response.success = False
            response.message = f'Error adjusting stiffness: {str(e)}'

        return response

    def emergency_stance_callback(self, goal_handle):
        """Action callback for emergency balance recovery"""
        self.get_logger().info('Initiating emergency balance recovery')

        # Stop normal locomotion
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

        # Move to stable stance
        cmd_msg = JointState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()

        # Define stable stance joint positions
        cmd_msg.name = ['left_hip_pitch', 'right_hip_pitch', 'left_knee', 'right_knee']
        cmd_msg.position = [-0.3, -0.3, 0.6, 0.6]  # Stable standing position

        # Execute the stance change
        for i in range(100):  # Simulate gradual movement over 1 second
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = EmergencyStance.Result()
                result.success = False
                result.message = 'Emergency stance canceled'
                return result

            self.joint_cmd_pub.publish(cmd_msg)

            # Provide feedback on progress
            feedback_msg = EmergencyStance.Feedback()
            feedback_msg.progress_percentage = (i + 1) / 100.0 * 100
            goal_handle.publish_feedback(feedback_msg)

            time.sleep(0.01)  # 10ms sleep

        goal_handle.succeed()
        result = EmergencyStance.Result()
        result.success = True
        result.message = 'Emergency stance achieved'

        return result

def main(args=None):
    rclpy.init(args=args)
    balance_controller = HumanoidBalanceController()

    try:
        rclpy.spin(balance_controller)
    except KeyboardInterrupt:
        pass
    finally:
        balance_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Quality of Service Considerations for Humanoid Robotics

When implementing communication patterns for humanoid robots, consider the appropriate QoS settings:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# For critical sensor data (IMU, encoders) - must be reliable
critical_sensor_qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE
)

# For high-frequency sensor data (camera, lidar) - may drop messages but low latency
high_freq_sensor_qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE
)

# For configuration parameters - must persist and be available to late-joining nodes
config_param_qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL
)

# For command messages - must be reliable to ensure safety
command_qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE
)
```

## Summary

ROS 2 provides three distinct communication patterns, each suited for different use cases in humanoid robotics:

- **Topics** provide asynchronous, publish-subscribe communication ideal for sensor data and status updates
- **Services** offer synchronous request-response communication for immediate queries and commands
- **Actions** enable long-running operations with feedback and cancellation capabilities

Understanding when to use each pattern is crucial for designing effective humanoid robotic systems. The choice of communication pattern affects system performance, reliability, and maintainability.

## Key Takeaways

- Topics are best for continuous data streams and broadcasting
- Services are ideal for immediate request-response interactions
- Actions are designed for complex, long-running tasks
- Quality of Service settings allow fine-tuning communication behavior
- Proper selection of communication patterns is essential for robust system design
- In humanoid robotics, safety-critical communications should use RELIABLE QoS policies
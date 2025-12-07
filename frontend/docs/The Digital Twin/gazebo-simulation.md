---
sidebar_position: 6
title: "Gazebo Physics Simulation for Humanoid Robots"
description: "Master Gazebo simulation for humanoid robots with accurate physics, sensors, and environments"
keywords: [Gazebo, physics simulation, humanoid robotics, ROS2, sensors, contacts, physics]
---

# Gazebo Physics Simulation for Humanoid Robots

Gazebo is a powerful physics simulation environment that plays a crucial role in humanoid robotics development. This chapter will guide you through setting up and configuring Gazebo for realistic humanoid robot simulation with accurate physics, sensor modeling, and environmental interactions.

## Learning Objectives

- Configure Gazebo for humanoid robot physics simulation
- Implement accurate contact modeling for bipedal locomotion
- Simulate realistic sensors (LiDAR, cameras, IMU, force/torque)
- Create diverse environments for testing humanoid capabilities
- Tune physics parameters for simulation accuracy

## Gazebo Architecture for Humanoid Robotics

Gazebo provides a comprehensive simulation environment with several key components relevant to humanoid robotics:

### Physics Engine
Gazebo uses one of several physics engines (ODE, Bullet, DART) to simulate the physical interactions of your humanoid robot:

- **ODE (Open Dynamics Engine)**: Good balance of performance and accuracy
- **Bullet**: High-performance physics with advanced collision detection
- **DART**: Advanced dynamics and real-time simulation capabilities

For humanoid robots, ODE is often the default choice, though DART may be preferred for more complex dynamics involving contact and balance.

### Sensor Simulation
Gazebo provides realistic simulation of various sensors commonly used in humanoid robots:

- **Camera sensors**: RGB, depth, stereo vision
- **LiDAR sensors**: 2D and 3D laser scanners
- **IMU sensors**: Accelerometer and gyroscope simulation
- **Force/Torque sensors**: Joint and contact force measurement
- **GPS sensors**: Position and velocity in global frame

### Rendering System
The rendering system provides visualization of the simulation with support for:

- **Realistic lighting**: Physically-based rendering
- **Material properties**: Accurate surface reflections and textures
- **Camera simulation**: Multiple viewpoints and sensor rendering

## Setting Up Gazebo for Humanoid Robots

### Installing Gazebo and ROS 2 Integration

First, ensure you have Gazebo and the ROS 2 integration packages installed:

```bash
# Install Gazebo Garden (or newer version)
sudo apt update
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control

# Install additional Gazebo plugins for humanoid simulation
sudo apt install ros-humble-ign-ros2-control ros-humble-ros2-control-test-assets
```

### Configuring Your Humanoid Robot for Gazebo

To properly simulate your humanoid robot in Gazebo, you need to add Gazebo-specific tags to your URDF. Here's an example of how to extend your humanoid robot model:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Include Gazebo plugins -->
  <gazebo>
    <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
      <parameters>$(find my_humanoid_description)/config/humanoid_controllers.yaml</parameters>
    </plugin>
  </gazebo>

  <!-- Example humanoid link with Gazebo properties -->
  <link name="left_foot">
    <inertial>
      <mass value="1.0" />
      <origin xyz="0.05 0 -0.02" />
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01" />
    </inertial>
    <visual>
      <origin xyz="0.05 0 -0.02" rpy="0 0 0" />
      <geometry>
        <box size="0.2 0.1 0.04" />
      </geometry>
      <material name="green" />
    </visual>
    <collision>
      <origin xyz="0.05 0 -0.02" rpy="0 0 0" />
      <geometry>
        <box size="0.2 0.1 0.04" />
      </geometry>
    </collision>
  </link>

  <!-- Gazebo-specific properties for the foot link -->
  <gazebo reference="left_foot">
    <!-- Contact properties for realistic foot-ground interaction -->
    <mu1>0.8</mu1>  <!-- Friction coefficient in primary direction -->
    <mu2>0.8</mu2>  <!-- Friction coefficient in secondary direction -->
    <kp>1000000.0</kp>  <!-- Contact stiffness -->
    <kd>100.0</kd>     <!-- Contact damping -->
    <min_depth>0.001</min_depth>  <!-- Penetration depth before contact force is applied -->
    <max_vel>100.0</max_vel>      <!-- Maximum contact correction velocity -->

    <!-- Material for visualization -->
    <material>Gazebo/Green</material>
  </gazebo>

  <!-- Example of sensor integration -->
  <gazebo reference="imu_link">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <visualize>false</visualize>
      <imu>
        <angular_velocity>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
              <bias_mean>0.0000075</bias_mean>
              <bias_stddev>0.0000008</bias_stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
              <bias_mean>0.0000075</bias_mean>
              <bias_stddev>0.0000008</bias_stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
              <bias_mean>0.0000075</bias_mean>
              <bias_stddev>0.0000008</bias_stddev>
            </noise>
          </z>
        </angular_velocity>
        <linear_acceleration>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
              <bias_mean>0.0000075</bias_mean>
              <bias_stddev>0.0000008</bias_stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
              <bias_mean>0.0000075</bias_mean>
              <bias_stddev>0.0000008</bias_stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
              <bias_mean>0.0000075</bias_mean>
              <bias_stddev>0.0000008</bias_stddev>
            </noise>
          </z>
        </linear_acceleration>
      </imu>
    </sensor>
  </gazebo>

</robot>
```

## Advanced Physics Configuration for Humanoid Robots

### Contact Modeling

Proper contact modeling is critical for humanoid robots, especially for balance and locomotion:

```xml
<!-- Contact sensor for foot-ground contact detection -->
<gazebo reference="left_foot">
  <sensor name="left_foot_contact_sensor" type="contact">
    <always_on>true</always_on>
    <update_rate>1000</update_rate>
    <contact>
      <collision>left_foot_collision</collision>
    </contact>
    <visualize>false</visualize>
  </sensor>

  <!-- Contact properties for realistic interaction -->
  <collision>
    <max_contacts>10</max_contacts>
    <surface>
      <friction>
        <ode>
          <mu>0.8</mu>
          <mu2>0.8</mu2>
          <slip1>0.0</slip1>
          <slip2>0.0</slip2>
          <fdir1>0 0 0</fdir1>
        </ode>
        <torsional>
          <coefficient>0.8</coefficient>
          <use_patch_radius>true</use_patch_radius>
          <surface_radius>0.01</surface_radius>
        </torsional>
      </friction>
      <bounce>
        <restitution_coefficient>0.01</restitution_coefficient>
        <threshold>100000</threshold>
      </bounce>
      <contact>
        <collide_without_contact>false</collide_without_contact>
        <collide_without_contact_bitmask>1</collide_without_contact_bitmask>
        <ode>
          <soft_cfm>0</soft_cfm>
          <soft_erp>0.2</soft_erp>
          <kp>1e+10</kp>
          <kd>1</kd>
          <max_vel>100.0</max_vel>
          <min_depth>0.001</min_depth>
        </ode>
      </contact>
    </surface>
  </collision>
</gazebo>
```

### Joint Configuration for Humanoid Joints

Humanoid robots have complex joints that require careful configuration:

```xml
<!-- Example of a humanoid hip joint with proper limits and dynamics -->
<joint name="left_hip_pitch" type="revolute">
  <parent link="torso" />
  <child link="left_thigh" />
  <origin xyz="0.0 -0.1 -0.1" rpy="0 0 0" />
  <axis xyz="1 0 0" />
  <limit lower="-1.57" upper="0.7" effort="100.0" velocity="5.0" />
  <dynamics damping="1.0" friction="0.1" />
</joint>

<!-- Gazebo-specific joint properties -->
<gazebo reference="left_hip_pitch">
  <provide_feedback>true</provide_feedback>
  <implicit_spring_damper>true</implicit_spring_damper>
  <axis>
    <dynamics>
      <damping>1.0</damping>
      <friction>0.1</friction>
      <spring_reference>0</spring_reference>
      <spring_stiffness>0</spring_stiffness>
    </dynamics>
  </axis>
</gazebo>
```

## Sensor Simulation for Humanoid Robots

### Camera Simulation

For humanoid robots, cameras are essential for perception:

```xml
<gazebo reference="head_camera">
  <sensor name="head_camera" type="camera">
    <always_on>true</always_on>
    <update_rate>30</update_rate>
    <camera>
      <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>head_camera_optical_frame</frame_name>
      <topic_name>image_raw</topic_name>
    </plugin>
  </sensor>
</gazebo>
```

### LiDAR Simulation

Many humanoid robots use LiDAR for environment perception:

```xml
<gazebo reference="lidar_mount">
  <sensor name="humanoid_lidar" type="ray">
    <always_on>true</always_on>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>10.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>laser</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
    </plugin>
  </sensor>
</gazebo>
```

## Creating Simulation Worlds

### Basic World File

Create a world file (`worlds/humanoid_lab.world`) for humanoid testing:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_lab">
    <!-- Include the Ground Plane model -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include the sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Add a simple humanoid lab environment -->
    <light name="ambient_light" type="directional">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>20</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
    </light>

    <!-- Add some obstacles for navigation testing -->
    <model name="obstacle_1">
      <pose>-1 0 0.5 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.2 0.2 1.0</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.2 0.2 1.0</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Add a ramp for walking challenge -->
    <model name="ramp">
      <pose>2 0 0 0 0.3 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.0 1.0 0.1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.0 1.0 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Add textured floor for better visual reference -->
    <model name="floor">
      <pose>0 0 -0.01 0 0 0</pose>
      <static>true</static>
      <link name="floor_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 10 0.02</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 10 0.02</size>
            </box>
          </geometry>
          <material>
            <script>
              <uri>file://media/materials/scripts/gazebo.material</uri>
              <name>Gazebo/GrassFloor</name>
            </script>
          </material>
        </visual>
      </link>
    </model>

  </world>
</sdf>
```

## Launching Humanoid Simulation

### Launch File for Simulation

Create a launch file (`launch/humanoid_gazebo.launch.py`) to start your humanoid simulation:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare(package='my_humanoid_description').find('my_humanoid_description')

    # Launch Arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_simulator = LaunchConfiguration('use_simulator')
    headless = LaunchConfiguration('headless')
    world = LaunchConfiguration('world')
    robot_name = LaunchConfiguration('robot_name')

    # Launch Arguments
    declare_use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true')

    declare_use_simulator_arg = DeclareLaunchArgument(
        'use_simulator',
        default_value='true',
        description='Whether to start the simulator')

    declare_headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Whether to execute gzclient headless')

    declare_world_arg = DeclareLaunchArgument(
        'world',
        description='Choose one of the world files from `/my_humanoid_description/worlds`',
        default_value=os.path.join(pkg_share, 'worlds', 'humanoid_lab.world'))

    declare_robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        description='Name of the robot',
        default_value='humanoid_robot')

    # Start Gazebo with the specified world
    start_gazebo_spawner_cmd = ExecuteProcess(
        cmd=['ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
             '-entity', robot_name,
             '-file', os.path.join(pkg_share, 'urdf', 'humanoid_robot.urdf'),
             '-x', '0', '-y', '0', '-z', '1.0'],
        output='screen')

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[{'use_sim_time': use_sim_time,
                    'robot_description': open(os.path.join(pkg_share, 'urdf', 'humanoid_robot.urdf')).read()}])

    # Joint State Publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[('/joint_states', 'joint_states')])

    # Controller Manager
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            os.path.join(pkg_share, 'config', 'humanoid_controllers.yaml'),
            {'use_sim_time': use_sim_time}
        ],
        remappings=[
            ('/controller_manager/robot_description', '/robot_description'),
        ],
        output='both'
    )

    # Spawn controllers
    spawn_joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'start',
             'joint_state_broadcaster'],
        output='screen'
    )

    spawn_base_leg_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'start',
             'base_leg_controller'],
        output='screen'
    )

    # Create launch description
    ld = LaunchDescription()

    # Declare launch arguments
    ld.add_action(declare_use_sim_time_arg)
    ld.add_action(declare_use_simulator_arg)
    ld.add_action(declare_headless_arg)
    ld.add_action(declare_world_arg)
    ld.add_action(declare_robot_name_arg)

    # Add nodes and processes
    ld.add_action(robot_state_publisher)
    ld.add_action(joint_state_publisher)
    ld.add_action(controller_manager)
    ld.add_action(spawn_joint_state_broadcaster)
    ld.add_action(spawn_base_leg_controller)
    ld.add_action(start_gazebo_spawner_cmd)

    return ld
```

## Physics Parameter Tuning

### Balancing Simulation Accuracy and Performance

For humanoid robots, finding the right balance between accuracy and performance is crucial:

```yaml
# physics_config.yaml
physics:
  type: ode  # Physics engine type
  max_step_size: 0.001  # Simulation step size (smaller = more accurate but slower)
  real_time_factor: 1.0  # Desired rate of simulation vs real time
  max_contacts: 20       # Maximum number of contacts between two geoms

  # ODE-specific parameters
  ode:
    solver_type: quick   # Solver type (quick or pgsl)
    iters: 1000          # Number of iterations in each solver loop
    sor: 1.3             # Successive Over Relaxation parameter

    # Constraints
    cfm: 0.0             # Global constraint force mixing parameter
    erp: 0.2             # Global error reduction parameter
    contact_surface_layer: 0.001  # Contact layer thickness
    contact_max_correcting_vel: 100.0  # Maximum contact correction velocity
```

## Advanced Simulation Techniques

### Ground Truth and Perception Simulation

For humanoid robots, accurate perception simulation is essential:

```xml
<!-- Ground truth for validation -->
<gazebo reference="humanoid_robot">
  <plugin name="gazebo_ros_p3d" filename="libgazebo_ros_p3d.so">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <body_name>base_link</body_name>
    <topic_name>ground_truth/state</topic_name>
    <gaussian_noise>0.01</gaussian_noise>
    <frame_name>map</frame_name>
  </plugin>
</gazebo>
```

### Multi-Robot Simulation

For humanoid teams or human-robot interaction:

```xml
<!-- Example of including multiple humanoid robots -->
<sdf version="1.7">
  <world name="multi_humanoid_world">
    <!-- First humanoid robot -->
    <include>
      <uri>model://humanoid_robot_1</uri>
      <name>humanoid_1</name>
      <pose>0 0 0 0 0 0</pose>
    </include>

    <!-- Second humanoid robot -->
    <include>
      <uri>model://humanoid_robot_2</uri>
      <name>humanoid_2</name>
      <pose>2 0 0 0 0 0</pose>
    </include>

    <!-- Shared environment elements -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>
  </world>
</sdf>
```

## Troubleshooting Common Issues

### 1. Robot Falls Through Floor
- Check collision geometries in URDF
- Verify contact parameters (mu1, mu2, kp, kd)
- Increase max_step_size or adjust solver parameters

### 2. Unstable Joint Behavior
- Verify inertial properties match physical reality
- Check joint limits and dynamics parameters
- Adjust controller gains if using PID controllers

### 3. Poor Performance
- Simplify collision geometries (use boxes/cylinders instead of meshes)
- Reduce update rates for sensors
- Use less complex physics engine if accuracy permits

### 4. Sensor Noise Issues
- Verify noise parameters match real sensors
- Check sensor update rates
- Ensure proper frame orientations

## Best Practices for Humanoid Simulation

### 1. Model Accuracy
- Use realistic inertial properties
- Match joint limits to physical robot
- Include actuator dynamics in simulation

### 2. Environment Design
- Create diverse environments for robust testing
- Include realistic friction and contact surfaces
- Add visual landmarks for perception testing

### 3. Validation
- Compare simulation and real robot behavior
- Validate sensor models against real hardware
- Test at various speeds and conditions

## Summary

Gazebo provides a powerful platform for humanoid robot simulation with accurate physics, realistic sensor modeling, and flexible environment creation. Proper configuration of contact properties, joint dynamics, and sensor models is essential for creating a useful digital twin of your humanoid robot.

The key to successful humanoid simulation lies in balancing accuracy with computational performance while ensuring that the simulated robot behaves similarly to the physical robot. This enables safe testing of control algorithms and perception systems before deployment on expensive hardware.
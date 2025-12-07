---
sidebar_position: 7
title: "URDF for Humanoid Robot Modeling"
description: "Creating and using URDF files to model humanoid robots for simulation and control"
keywords: [URDF, Humanoid Robot, Robot Modeling, Simulation, ROS2, Xacro, Gazebo]
---

# URDF for Humanoid Robot Modeling

This chapter covers the Unified Robot Description Format (URDF) and its extensions for modeling humanoid robots. URDF is essential for creating accurate robot models that can be used in simulation, visualization, and control systems.

## Learning Objectives

- Understand URDF fundamentals and XML structure
- Learn to create kinematic chains for humanoid robots
- Master joint types and their applications in humanoid modeling
- Implement sensor mounting and visualization elements
- Use Xacro for parameterized and modular robot descriptions
- Validate URDF models for humanoid applications

## Introduction to URDF

The Unified Robot Description Format (URDF) is an XML-based format used to describe robot models in ROS. For humanoid robots, URDF provides the essential framework for defining the robot's physical structure, kinematic relationships, and geometric properties.

### URDF Structure for Humanoid Robots

A humanoid robot's URDF typically includes:

- **Links**: Rigid bodies (limbs, torso, head)
- **Joints**: Connections between links (hips, knees, ankles, shoulders, elbows, wrists)
- **Materials**: Visual appearance properties
- **Gazebo plugins**: Simulation-specific extensions
- **Transmission elements**: Motor control interfaces

## Basic URDF Elements

### Links

Links represent rigid bodies in the robot. For humanoid robots, these include limbs, torso, and head segments:

```xml
<link name="base_link">
  <inertial>
    <mass value="10.0" />
    <origin xyz="0 0 0" />
    <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1" />
  </inertial>
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0" />
    <geometry>
      <mesh filename="package://humanoid_description/meshes/base_link.dae" />
    </geometry>
    <material name="blue">
      <color rgba="0 0 1 1" />
    </material>
  </visual>
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0" />
    <geometry>
      <mesh filename="package://humanoid_description/meshes/base_link_collision.dae" />
    </geometry>
  </collision>
</link>
```

### Inertial Properties

For humanoid robots, accurate inertial properties are critical for stable simulation and control:

```xml
<inertial>
  <!-- Mass in kg -->
  <mass value="2.5" />
  <!-- Origin of the inertial reference frame relative to the link origin -->
  <origin xyz="0.0 0.0 0.1" rpy="0 0 0" />
  <!-- Inertia matrix (symmetric, so only 6 values needed) -->
  <inertia
    ixx="0.01" ixy="0.0" ixz="0.0"
    iyy="0.02" iyz="0.0"
    izz="0.015" />
</inertial>
```

The inertia values should be calculated based on the actual physical properties of the robot segment. For a humanoid leg segment, you might use simplified geometric approximations:

- **Cylinder approximation**: For limbs `Ixx = Iyy = (1/4)*m*r² + (1/12)*m*L²`, `Izz = (1/2)*m*r²`
- **Box approximation**: For torso segments `Ixx = (1/12)*m*(h² + d²)`, etc.

### Visual and Collision Elements

Visual elements define how the robot appears in RViz and simulation:

```xml
<visual>
  <origin xyz="0 0 0" rpy="0 0 0" />
  <geometry>
    <mesh filename="package://humanoid_description/meshes/leg_segment.stl" />
  </geometry>
  <material name="gray">
    <color rgba="0.5 0.5 0.5 1.0" />
  </material>
</visual>
```

Collision elements define the physical collision boundaries:

```xml
<collision>
  <origin xyz="0 0 0" rpy="0 0 0" />
  <geometry>
    <!-- Use simplified geometry for collision to improve performance -->
    <cylinder length="0.4" radius="0.05" />
  </geometry>
</collision>
```

## Joint Types in Humanoid Robots

Humanoid robots use various joint types to replicate human-like movement:

### Revolute Joints

Revolute joints allow rotation around a single axis, commonly used for:
- Hip joints (pitch, roll, yaw)
- Knee joints (pitch only)
- Ankle joints (pitch, roll)
- Shoulder joints (pitch, roll, yaw)
- Elbow joints (pitch only)
- Wrist joints (pitch, roll)

```xml
<joint name="left_knee_joint" type="revolute">
  <parent link="left_thigh" />
  <child link="left_shin" />
  <origin xyz="0 0 -0.4" rpy="0 0 0" />
  <axis xyz="1 0 0" />
  <limit lower="-2.0" upper="0.5" effort="100.0" velocity="5.0" />
  <dynamics damping="1.0" friction="0.1" />
</joint>
```

### Continuous Joints

Continuous joints allow unlimited rotation around an axis, used for:
- Waist rotation
- Head rotation

```xml
<joint name="waist_yaw_joint" type="continuous">
  <parent link="pelvis" />
  <child link="torso" />
  <origin xyz="0 0 0.15" rpy="0 0 0" />
  <axis xyz="0 0 1" />
  <dynamics damping="2.0" friction="0.2" />
</joint>
```

### Fixed Joints

Fixed joints connect links without allowing relative motion, used for:
- Attaching sensors
- Connecting accessories

```xml
<joint name="imu_mount" type="fixed">
  <parent link="torso" />
  <child link="imu_link" />
  <origin xyz="0.1 0 0.2" rpy="0 0 0" />
</joint>

<link name="imu_link">
  <inertial>
    <mass value="0.01" />
    <origin xyz="0 0 0" />
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001" />
  </inertial>
</link>
```

## Complete Humanoid Robot URDF

Here's a simplified example of a humanoid robot URDF:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Materials -->
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="green">
    <color rgba="0.0 0.8 0.0 1.0"/>
  </material>
  <material name="grey">
    <color rgba="0.5 0.5 0.5 1.0"/>
  </material>
  <material name="orange">
    <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>
  <material name="brown">
    <color rgba="0.870588235294 0.811764705882 0.764705882353 1.0"/>
  </material>
  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0" />
      <origin xyz="0 0 0" />
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <box size="0.1 0.1 0.1" />
      </geometry>
      <material name="white" />
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <box size="0.1 0.1 0.1" />
      </geometry>
    </collision>
  </link>

  <!-- Torso -->
  <link name="torso">
    <inertial>
      <mass value="8.0" />
      <origin xyz="0 0 0.2" />
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.1" />
    </inertial>
    <visual>
      <origin xyz="0 0 0.2" rpy="0 0 0" />
      <geometry>
        <box size="0.2 0.2 0.4" />
      </geometry>
      <material name="orange" />
    </visual>
    <collision>
      <origin xyz="0 0 0.2" rpy="0 0 0" />
      <geometry>
        <box size="0.2 0.2 0.4" />
      </geometry>
    </collision>
  </link>

  <joint name="torso_joint" type="fixed">
    <parent link="base_link" />
    <child link="torso" />
    <origin xyz="0 0 0.05" />
  </joint>

  <!-- Head -->
  <link name="head">
    <inertial>
      <mass value="2.0" />
      <origin xyz="0 0 0.05" />
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01" />
    </inertial>
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0" />
      <geometry>
        <sphere radius="0.1" />
      </geometry>
      <material name="white" />
    </visual>
    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0" />
      <geometry>
        <sphere radius="0.1" />
      </geometry>
    </collision>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="torso" />
    <child link="head" />
    <origin xyz="0 0 0.4" />
    <axis xyz="0 1 0" />
    <limit lower="-1.0" upper="1.0" effort="10.0" velocity="2.0" />
  </joint>

  <!-- Left Arm -->
  <link name="left_shoulder">
    <inertial>
      <mass value="1.0" />
      <origin xyz="0 0 -0.05" />
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001" />
    </inertial>
    <visual>
      <origin xyz="0 0 -0.05" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.1" radius="0.05" />
      </geometry>
      <material name="blue" />
    </visual>
    <collision>
      <origin xyz="0 0 -0.05" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.1" radius="0.05" />
      </geometry>
    </collision>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso" />
    <child link="left_shoulder" />
    <origin xyz="0.15 0 0.3" rpy="0 0 0" />
    <axis xyz="0 1 0" />
    <limit lower="-1.57" upper="1.57" effort="20.0" velocity="3.0" />
  </joint>

  <link name="left_upper_arm">
    <inertial>
      <mass value="1.5" />
      <origin xyz="0 0 -0.1" />
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.001" />
    </inertial>
    <visual>
      <origin xyz="0 0 -0.1" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.2" radius="0.04" />
      </geometry>
      <material name="blue" />
    </visual>
    <collision>
      <origin xyz="0 0 -0.1" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.2" radius="0.04" />
      </geometry>
    </collision>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_shoulder" />
    <child link="left_upper_arm" />
    <origin xyz="0 0 -0.1" rpy="0 0 0" />
    <axis xyz="1 0 0" />
    <limit lower="-2.0" upper="0.5" effort="20.0" velocity="3.0" />
  </joint>

  <link name="left_lower_arm">
    <inertial>
      <mass value="1.0" />
      <origin xyz="0 0 -0.1" />
      <inertia ixx="0.003" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.001" />
    </inertial>
    <visual>
      <origin xyz="0 0 -0.1" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.2" radius="0.035" />
      </geometry>
      <material name="blue" />
    </visual>
    <collision>
      <origin xyz="0 0 -0.1" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.2" radius="0.035" />
      </geometry>
    </collision>
  </link>

  <joint name="left_wrist_joint" type="revolute">
    <parent link="left_upper_arm" />
    <child link="left_lower_arm" />
    <origin xyz="0 0 -0.2" rpy="0 0 0" />
    <axis xyz="1 0 0" />
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="3.0" />
  </joint>

  <!-- Left Leg -->
  <link name="left_hip">
    <inertial>
      <mass value="2.0" />
      <origin xyz="0 0 -0.1" />
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.005" />
    </inertial>
    <visual>
      <origin xyz="0 0 -0.1" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.2" radius="0.06" />
      </geometry>
      <material name="green" />
    </visual>
    <collision>
      <origin xyz="0 0 -0.1" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.2" radius="0.06" />
      </geometry>
    </collision>
  </link>

  <joint name="left_hip_joint" type="revolute">
    <parent link="torso" />
    <child link="left_hip" />
    <origin xyz="0.05 0.1 -0.1" rpy="0 0 0" />
    <axis xyz="1 0 0" />
    <limit lower="-1.57" upper="0.7" effort="50.0" velocity="2.0" />
  </joint>

  <link name="left_thigh">
    <inertial>
      <mass value="3.0" />
      <origin xyz="0 0 -0.2" />
      <inertia ixx="0.04" ixy="0" ixz="0" iyy="0.04" iyz="0" izz="0.01" />
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.4" radius="0.05" />
      </geometry>
      <material name="green" />
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.4" radius="0.05" />
      </geometry>
    </collision>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_hip" />
    <child link="left_thigh" />
    <origin xyz="0 0 -0.2" rpy="0 0 0" />
    <axis xyz="1 0 0" />
    <limit lower="-2.0" upper="0.5" effort="50.0" velocity="2.0" />
  </joint>

  <link name="left_shin">
    <inertial>
      <mass value="2.5" />
      <origin xyz="0 0 -0.2" />
      <inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.01" />
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.4" radius="0.045" />
      </geometry>
      <material name="green" />
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.4" radius="0.045" />
      </geometry>
    </collision>
  </link>

  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_thigh" />
    <child link="left_shin" />
    <origin xyz="0 0 -0.4" rpy="0 0 0" />
    <axis xyz="1 0 0" />
    <limit lower="-0.5" upper="0.5" effort="30.0" velocity="2.0" />
  </joint>

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

  <joint name="left_foot_joint" type="fixed">
    <parent link="left_shin" />
    <child link="left_foot" />
    <origin xyz="0 0 -0.4" rpy="0 0 0" />
  </joint>

  <!-- Right Leg (mirror of left leg) -->
  <link name="right_hip">
    <inertial>
      <mass value="2.0" />
      <origin xyz="0 0 -0.1" />
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.005" />
    </inertial>
    <visual>
      <origin xyz="0 0 -0.1" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.2" radius="0.06" />
      </geometry>
      <material name="green" />
    </visual>
    <collision>
      <origin xyz="0 0 -0.1" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.2" radius="0.06" />
      </geometry>
    </collision>
  </link>

  <joint name="right_hip_joint" type="revolute">
    <parent link="torso" />
    <child link="right_hip" />
    <origin xyz="0.05 -0.1 -0.1" rpy="0 0 0" />
    <axis xyz="1 0 0" />
    <limit lower="-1.57" upper="0.7" effort="50.0" velocity="2.0" />
  </joint>

  <link name="right_thigh">
    <inertial>
      <mass value="3.0" />
      <origin xyz="0 0 -0.2" />
      <inertia ixx="0.04" ixy="0" ixz="0" iyy="0.04" iyz="0" izz="0.01" />
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.4" radius="0.05" />
      </geometry>
      <material name="green" />
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.4" radius="0.05" />
      </geometry>
    </collision>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_hip" />
    <child link="right_thigh" />
    <origin xyz="0 0 -0.2" rpy="0 0 0" />
    <axis xyz="1 0 0" />
    <limit lower="-2.0" upper="0.5" effort="50.0" velocity="2.0" />
  </joint>

  <link name="right_shin">
    <inertial>
      <mass value="2.5" />
      <origin xyz="0 0 -0.2" />
      <inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.01" />
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.4" radius="0.045" />
      </geometry>
      <material name="green" />
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.4" radius="0.045" />
      </geometry>
    </collision>
  </link>

  <joint name="right_ankle_joint" type="revolute">
    <parent link="right_thigh" />
    <child link="right_shin" />
    <origin xyz="0 0 -0.4" rpy="0 0 0" />
    <axis xyz="1 0 0" />
    <limit lower="-0.5" upper="0.5" effort="30.0" velocity="2.0" />
  </joint>

  <link name="right_foot">
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

  <joint name="right_foot_joint" type="fixed">
    <parent link="right_shin" />
    <child link="right_foot" />
    <origin xyz="0 0 -0.4" rpy="0 0 0" />
  </joint>

  <!-- Sensors -->
  <joint name="imu_joint" type="fixed">
    <parent link="torso" />
    <child link="imu_link" />
    <origin xyz="0.0 0.0 0.1" rpy="0 0 0" />
  </joint>

  <link name="imu_link" />

  <!-- Gazebo plugins -->
  <gazebo reference="base_link">
    <material>Gazebo/White</material>
  </gazebo>

  <gazebo reference="torso">
    <material>Gazebo/Orange</material>
  </gazebo>

  <gazebo reference="head">
    <material>Gazebo/White</material>
  </gazebo>

  <gazebo reference="left_shoulder">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="left_upper_arm">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="left_lower_arm">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="left_hip">
    <material>Gazebo/Green</material>
  </gazebo>

  <gazebo reference="left_thigh">
    <material>Gazebo/Green</material>
  </gazebo>

  <gazebo reference="left_shin">
    <material>Gazebo/Green</material>
  </gazebo>

  <gazebo reference="left_foot">
    <material>Gazebo/Green</material>
  </gazebo>

  <gazebo reference="right_hip">
    <material>Gazebo/Green</material>
  </gazebo>

  <gazebo reference="right_thigh">
    <material>Gazebo/Green</material>
  </gazebo>

  <gazebo reference="right_shin">
    <material>Gazebo/Green</material>
  </gazebo>

  <gazebo reference="right_foot">
    <material>Gazebo/Green</material>
  </gazebo>

  <!-- Transmission elements for ros2_control -->
  <transmission name="left_hip_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_hip_joint">
      <hardwareInterface>PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_hip_motor">
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="left_knee_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_knee_joint">
      <hardwareInterface>PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_knee_motor">
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="left_ankle_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_ankle_joint">
      <hardwareInterface>PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_ankle_motor">
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="right_hip_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_hip_joint">
      <hardwareInterface>PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_hip_motor">
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="right_knee_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_knee_joint">
      <hardwareInterface>PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_knee_motor">
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="right_ankle_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_ankle_joint">
      <hardwareInterface>PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_ankle_motor">
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

</robot>
```

## Using Xacro for Modular Robot Descriptions

Xacro (XML Macros) extends URDF with macros, properties, and mathematical expressions, making complex humanoid robot descriptions more manageable:

### Basic Xacro Structure

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_robot">

  <!-- Define constants -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="torso_length" value="0.5" />
  <xacro:property name="leg_length" value="0.8" />
  <xacro:property name="arm_length" value="0.6" />

  <!-- Define materials -->
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>

  <!-- Macro for creating a simple limb -->
  <xacro:macro name="simple_limb" params="name parent xyz_origin axis_xyz lower_limit:=0 upper_limit:=0 effort:=100 velocity:=5">
    <link name="${name}_link">
      <inertial>
        <mass value="1.0" />
        <origin xyz="0 0 -0.1" />
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.005" />
      </inertial>
      <visual>
        <origin xyz="0 0 -0.1" rpy="0 0 0" />
        <geometry>
          <cylinder length="0.2" radius="0.05" />
        </geometry>
        <material name="black" />
      </visual>
      <collision>
        <origin xyz="0 0 -0.1" rpy="0 0 0" />
        <geometry>
          <cylinder length="0.2" radius="0.05" />
        </geometry>
      </collision>
    </link>

    <joint name="${name}_joint" type="revolute">
      <parent link="${parent}" />
      <child link="${name}_link" />
      <origin xyz="${xyz_origin}" rpy="0 0 0" />
      <axis xyz="${axis_xyz}" />
      <limit lower="${lower_limit}" upper="${upper_limit}" effort="${effort}" velocity="${velocity}" />
    </joint>
  </xacro:macro>

  <!-- Use the macro to create joints -->
  <xacro:simple_limb name="left_knee" parent="left_hip"
                     xyz_origin="0 0 -0.2" axis_xyz="1 0 0"
                     lower_limit="-2.0" upper_limit="0.5" />

  <xacro:simple_limb name="right_knee" parent="right_hip"
                     xyz_origin="0 0 -0.2" axis_xyz="1 0 0"
                     lower_limit="-2.0" upper_limit="0.5" />

</robot>
```

### Advanced Xacro for Humanoid Body Parts

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Leg macro with complete kinematic chain -->
  <xacro:macro name="humanoid_leg" params="side parent_link position_offset hip_limits knee_limits ankle_limits">
    <!-- Hip joint -->
    <joint name="${side}_hip_joint" type="revolute">
      <parent link="${parent_link}" />
      <child link="${side}_hip" />
      <origin xyz="${position_offset}" rpy="0 0 0" />
      <axis xyz="1 0 0" />
      <limit lower="${hip_limits[0]}" upper="${hip_limits[1]}" effort="100.0" velocity="2.0" />
    </joint>

    <link name="${side}_hip">
      <inertial>
        <mass value="2.0" />
        <origin xyz="0 0 -0.1" />
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.005" />
      </inertial>
      <visual>
        <origin xyz="0 0 -0.1" rpy="0 0 0" />
        <geometry>
          <cylinder length="0.2" radius="0.06" />
        </geometry>
        <material name="${side}_color" />
      </visual>
      <collision>
        <origin xyz="0 0 -0.1" rpy="0 0 0" />
        <geometry>
          <cylinder length="0.2" radius="0.06" />
        </geometry>
      </collision>
    </link>

    <!-- Thigh -->
    <joint name="${side}_knee_joint" type="revolute">
      <parent link="${side}_hip" />
      <child link="${side}_thigh" />
      <origin xyz="0 0 -0.2" rpy="0 0 0" />
      <axis xyz="1 0 0" />
      <limit lower="${knee_limits[0]}" upper="${knee_limits[1]}" effort="100.0" velocity="2.0" />
    </joint>

    <link name="${side}_thigh">
      <inertial>
        <mass value="3.0" />
        <origin xyz="0 0 -0.2" />
        <inertia ixx="0.04" ixy="0" ixz="0" iyy="0.04" iyz="0" izz="0.01" />
      </inertial>
      <visual>
        <origin xyz="0 0 -0.2" rpy="0 0 0" />
        <geometry>
          <cylinder length="0.4" radius="0.05" />
        </geometry>
        <material name="${side}_color" />
      </visual>
      <collision>
        <origin xyz="0 0 -0.2" rpy="0 0 0" />
        <geometry>
          <cylinder length="0.4" radius="0.05" />
        </geometry>
      </collision>
    </link>

    <!-- Shin -->
    <joint name="${side}_ankle_joint" type="revolute">
      <parent link="${side}_thigh" />
      <child link="${side}_shin" />
      <origin xyz="0 0 -0.4" rpy="0 0 0" />
      <axis xyz="1 0 0" />
      <limit lower="${ankle_limits[0]}" upper="${ankle_limits[1]}" effort="50.0" velocity="2.0" />
    </joint>

    <link name="${side}_shin">
      <inertial>
        <mass value="2.5" />
        <origin xyz="0 0 -0.15" />
        <inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.01" />
      </inertial>
      <visual>
        <origin xyz="0 0 -0.15" rpy="0 0 0" />
        <geometry>
          <cylinder length="0.3" radius="0.045" />
        </geometry>
        <material name="${side}_color" />
      </visual>
      <collision>
        <origin xyz="0 0 -0.15" rpy="0 0 0" />
        <geometry>
          <cylinder length="0.3" radius="0.045" />
        </geometry>
      </collision>
    </link>

    <!-- Foot -->
    <joint name="${side}_foot_joint" type="fixed">
      <parent link="${side}_shin" />
      <child link="${side}_foot" />
      <origin xyz="0 0 -0.3" rpy="0 0 0" />
    </joint>

    <link name="${side}_foot">
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
        <material name="${side}_color" />
      </visual>
      <collision>
        <origin xyz="0.05 0 -0.02" rpy="0 0 0" />
        <geometry>
          <box size="0.2 0.1 0.04" />
        </geometry>
      </collision>
    </link>
  </xacro:macro>

  <!-- Material definitions -->
  <material name="left_color">
    <color rgba="0.0 0.8 0.0 1.0" />
  </material>
  <material name="right_color">
    <color rgba="0.8 0.0 0.0 1.0" />
  </material>

  <!-- Use the leg macro to create both legs -->
  <xacro:humanoid_leg
    side="left"
    parent_link="torso"
    position_offset="0.05 0.1 -0.1"
    hip_limits="-1.57 0.7"
    knee_limits="-2.0 0.5"
    ankle_limits="-0.5 0.5" />

  <xacro:humanoid_leg
    side="right"
    parent_link="torso"
    position_offset="0.05 -0.1 -0.1"
    hip_limits="-1.57 0.7"
    knee_limits="-2.0 0.5"
    ankle_limits="-0.5 0.5" />

</robot>
```

## URDF Validation and Testing

### Validation Tools

Before using your URDF in simulation or on a real robot, validate it:

```bash
# Install urdfdom tools
sudo apt install liburdfdom-tools

# Validate URDF syntax
check_urdf /path/to/your/robot.urdf

# Print robot information
urdf_to_graphiz /path/to/your/robot.urdf
```

### Testing in RViz

Launch your robot model in RViz to visualize it:

```bash
# Launch robot state publisher
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:=$(cat robot.urdf)
```

### Kinematic Validation

For humanoid robots, it's important to validate the kinematic chain:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
import tf_transformations

class URDFValidator(Node):
    def __init__(self):
        super().__init__('urdf_validator')

        # Subscribe to joint states
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)

        # Initialize TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Known joint limits for validation
        self.joint_limits = {
            'left_hip_joint': (-1.57, 0.7),
            'left_knee_joint': (-2.0, 0.5),
            'left_ankle_joint': (-0.5, 0.5),
            'right_hip_joint': (-1.57, 0.7),
            'right_knee_joint': (-2.0, 0.5),
            'right_ankle_joint': (-0.5, 0.5),
        }

    def joint_callback(self, msg):
        """Validate incoming joint states"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                position = msg.position[i]

                # Check if joint is in our known limits
                if name in self.joint_limits:
                    min_limit, max_limit = self.joint_limits[name]

                    if position < min_limit or position > max_limit:
                        self.get_logger().warn(
                            f'Joint {name} position {position} outside limits '
                            f'[{min_limit}, {max_limit}]'
                        )

                # Check for extreme velocities (possible sensor errors)
                if i < len(msg.velocity) and abs(msg.velocity[i]) > 10.0:
                    self.get_logger().warn(f'Unusually high velocity for {name}: {msg.velocity[i]}')

def main():
    rclpy.init()
    validator = URDFValidator()

    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        pass
    finally:
        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Humanoid URDF

### 1. Proper Inertial Properties
- Use CAD software to calculate accurate inertial properties
- Verify that center of mass is physically plausible
- Ensure inertia matrices are positive definite

### 2. Realistic Joint Limits
- Base limits on physical robot capabilities
- Consider safety margins in software limits
- Account for soft limits in addition to hard limits

### 3. Efficient Collision Geometry
- Use simple shapes (boxes, cylinders, spheres) for collision
- Balance accuracy with computational efficiency
- Separate visual and collision geometries appropriately

### 4. Consistent Naming Conventions
- Use descriptive names that indicate function and position
- Follow consistent patterns (e.g., `left_hip_joint`, `right_knee_joint`)
- Consider left/right symmetry in naming

### 5. Modularity with Xacro
- Break down complex robots into reusable components
- Use parameters for sizing and configuration
- Create libraries of common components

## Simulation Considerations

### Gazebo-Specific Elements

Add Gazebo-specific elements to your URDF for proper simulation:

```xml
<gazebo>
  <!-- Prevent robot from falling through ground -->
  <static>false</static>

  <!-- Self-collision settings -->
  <self_collide>true</self_collide>

  <!-- Turn off gravity for specific links if needed -->
  <gravity>false</gravity>
</gazebo>

<!-- For joints with actuators -->
<gazebo reference="left_knee_joint">
  <implicit_spring_damper>1</implicit_spring_damper>
</gazebo>

<!-- For links with sensors -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>true</visualize>
  </sensor>
</gazebo>
```

## Summary

URDF is fundamental to humanoid robotics development, providing the essential bridge between the physical robot and its digital representation. Creating accurate, well-structured URDF models is crucial for:

- Simulation accuracy and stability
- Proper visualization in RViz and Gazebo
- Accurate kinematic and dynamic calculations
- Successful integration with ROS 2 control systems

The combination of URDF with Xacro macros allows for modular, maintainable robot descriptions that can be adapted for different humanoid robot configurations while maintaining consistency and reducing errors.

## Key Takeaways

- URDF describes robot structure with links, joints, and properties
- Xacro macros enable modular and parameterized robot descriptions
- Proper inertial properties are essential for simulation accuracy
- Joint limits should reflect physical robot constraints
- Collision geometry should balance accuracy with performance
- Gazebo-specific elements enhance simulation behavior
- Validation tools help catch errors before deployment
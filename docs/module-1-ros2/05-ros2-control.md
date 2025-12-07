---
title: "ROS 2 Control Basics"
description: "Understanding ROS 2 control for robotic systems"
slug: /module-1-ros2/05-ros2-control
tags: [ROS2, Control, Robotics, Controllers]
---

# ROS 2 Control Basics

This chapter introduces the ROS 2 Control framework, which provides a standardized way to interface with robot hardware controllers. This is essential for commanding actuators and reading sensor data on real robots.

## Learning Objectives

- Understand the ROS 2 Control architecture
- Learn about hardware interfaces and controllers
- Configure and launch controllers for a robot
- Implement basic control strategies

## ROS 2 Control Architecture

ROS 2 Control is a real-time capable controller framework that allows users to write controller code that can be loaded and unloaded at runtime. It provides a hardware abstraction layer that connects ROS 2 with the physical robot.
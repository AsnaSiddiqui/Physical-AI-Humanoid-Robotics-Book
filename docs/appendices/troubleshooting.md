---
title: "Troubleshooting Guide"
description: "Common issues and solutions for the Physical AI & Humanoid Robotics projects"
slug: /appendices/troubleshooting
tags: [Troubleshooting, FAQ, Issues, Solutions]
---

# Troubleshooting Guide

This guide provides solutions to common issues encountered when working with ROS 2, Gazebo, Isaac Sim, and other tools covered in this book.

## Common ROS 2 Issues

### Problem: Nodes not communicating across machines
**Solution**: Ensure all machines are on the same network and have the same `ROS_DOMAIN_ID` set.

### Problem: Package not found
**Solution**: Source the ROS 2 environment with `source /opt/ros/<distro>/setup.bash` and rebuild with `colcon build`.
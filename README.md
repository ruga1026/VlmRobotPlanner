# VlmRobotPlanner

VlmRobotPlanner is a demo project for Jetson Orin–based RC cars that combines:

- ROS 2 Humble
- Navigation2 (Nav2)
- RTAB-Map (rtabmap_ros)
- YOLO-based perception
- An Ollama-based Vision-Language Model (VLM)

The goal is to run autonomous navigation while interacting with the robot via natural-language prompts.

Received the Encouragement Award at the 7th National Undergraduate Capstone Design Competition organized 
by the Society for Aerospace System Engineering (SASE), August 2025.

Presentation Poster(in Korean):

<img width="442" height="640" alt="Image" src="https://github.com/user-attachments/assets/b4bd2c04-dde0-47bf-b3df-6dd5c73bcf0b" />

---

## 1. Hardware & Software

- **Boards**
  - NVIDIA Jetson Orin NX
  - NVIDIA Jetson Orin Nano Super
- **OS / Middleware**
  - Ubuntu 22.04 (JetPack-based)
  - ROS 2 Humble
- **Core stacks**
  - Nav2 (Navigation2)
  - RTAB-Map (rtabmap_ros)
  - YOLO (object detection)
  - Ollama VLM (multimodal LLM)

---

## 2. Repository Layout

At the top level the repository contains separate source trees for each Jetson board:

```text
VlmRobotPlanner/
 ├─ src_nx/      # ROS 2 packages for Jetson Orin NX
 └─ src_nano/    # ROS 2 packages for Jetson Orin Nano Super
````

Inside each `src_*` directory you will find the ROS 2 packages used in this project (e.g. PWM control, vehicle model with Nav2 launch files, YOLO, VLM, etc.).
Choose the directory that matches your board and use it as the source for your ROS 2 workspace.

---

## 3. Setup & Build

### 3.1 Create a ROS 2 workspace and clone

```bash
# Create workspace
mkdir -p ~/vlm_ws/src
cd ~/vlm_ws/src

# Clone this repository
git clone https://github.com/ruga1026/VlmRobotPlanner.git
```

### 3.2 Select board-specific sources

Pick one of the source trees depending on the Jetson board you are using (example for Orin NX):

```bash
# Example: use src_nx for Jetson Orin NX
cp -r VlmRobotPlanner/src_nx/* .

cd ~/vlm_ws
colcon build
source install/setup.bash
```

> Adjust paths or copy method as you prefer (symlink, overlay, etc.).
> The important part is that the packages from `src_nx` or `src_nano` end up inside your ROS 2 workspace `src/`.

---

## 4. Install Nav2 and RTAB-Map

### 4.1 Nav2 (Navigation2)

On ROS 2 Humble you can install Nav2 and its bringup package via apt:

```bash
sudo apt update
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
```

### 4.2 RTAB-Map (rtabmap_ros)

Install RTAB-Map for ROS 2 Humble (use source if there are issues on Jetson):

```bash
sudo apt install ros-humble-rtabmap-ros
```

If you encounter problems on Jetson (arm64), refer to the official `rtabmap_ros` documentation for building from source.

---

## 5. Ollama & VLM Model Configuration

This project uses **Ollama** to run a local vision-language model (VLM).

### 5.1 Install Ollama (summary)

On the Jetson, install Ollama (refer to Ollama/Jetson docs for details and any Jetson-specific notes):

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Then pull the multimodal model you want to use, for example:

```bash
# Example models (replace with the one you want)
ollama pull llava
# or
ollama pull llama3.2-vision
```

### 5.2 Configure model name in `vlm_node.py`

Inside the **`vlm`** package you will find `vlm_node.py`.
Edit this file and change the **model name** used when calling Ollama to match the model you actually pulled.

For example, if the code currently uses:

```python
model_name = "llava"
```

and you pulled `llama3.2-vision`, update it to:

```python
model_name = "llama3.2-vision"
```

> The exact variable or field name might differ; open `vlm_node.py` and modify the model-related string accordingly.

---

## 6. Running Vehicle PWM Control & Nav2

Make sure you have sourced both ROS 2 Humble and your workspace:

```bash
source /opt/ros/humble/setup.bash
source ~/vlm_ws/install/setup.bash
```

### 6.1 Start PWM control

This node publishes the PWM commands for throttle/steering:

```bash
ros2 launch pwm_control pwm_pub.launch.py
```

### 6.2 Start Nav2 for the vehicle

Launch Nav2 using the vehicle-specific launch file:

```bash
ros2 launch traxxas traxxas_nav2.launch.py
```

This will bring up Nav2 for the Traxxas-based vehicle configuration.
For details (map, TF frames, sensor topics, parameters), check the launch and config files inside the relevant package.

---

## 7. Running YOLO & VLM

After PWM control and Nav2 are running, start perception and the VLM.

### 7.1 Start **all** YOLO nodes in the `yolo` package

The YOLO package may contain multiple nodes (e.g., camera/image node, detection node, post-processing, etc.).
**Start all nodes provided by the `yolo` package** so that detection results are fully available to the rest of the system.

Depending on how the package is structured, this might be done by:

* A single launch file that starts every YOLO-related node, or
* Running several `ros2 run` / `ros2 launch` commands, one for each node.


Check the `yolo` package’s `launch/` and `src/` directories for the exact node and launch file names.

### 7.2 Start the VLM node

Next, launch the VLM node from the `vlm` package:

```bash
ros2 run vlm vlm_node
```

Again, adjust the command to match your actual package/launch layout.

### 7.3 Send prompts to the VLM

Once the VLM node is running and connected to:

* YOLO outputs (detections),
* Nav2 (map, pose, costmap, etc.),

you can interact with the system by sending prompts to the VLM.
The interaction interface (topic, service, action, or CLI input) is defined in `vlm_node.py` and/or the corresponding launch/config files.

Typical usage could be:

* “Avoid the obstacle in front and move to the goal on the right.”
* “Follow the detected person while staying inside the corridor.”
* “Describe what you see and suggest a path to the nearest exit.”

Refer to the node’s implementation for the exact ROS interface.

---

## 8. Recommended Run Sequence

A typical run sequence on the robot:

1. Source ROS 2 and workspace:

   ```bash
   source /opt/ros/humble/setup.bash
   source ~/vlm_ws/install/setup.bash
   ```
2. Start PWM control:

   ```bash
   ros2 launch pwm_control pwm_pub.launch.py
   ```
3. Start Nav2:

   ```bash
   ros2 launch traxxas traxxas_nav2.launch.py
   ```
4. Start **all YOLO nodes** in the `yolo` package.
5. Start the VLM node from the `vlm` package.
6. Send prompts to the VLM and observe the robot’s behavior.

---

## 9. Notes & Further Customization

* **Board selection**
  Use `src_nx` for Jetson Orin NX and `src_nano` for Jetson Orin Nano Super, not both at the same time.
* **Topic / frame names**
  Adapt camera, LiDAR, TF frame names, and Nav2 configuration to match your actual hardware.
* **Model experiments**
  You can easily try different VLM models by:

  1. `ollama pull <model>`
  2. Updating the model name in `vlm_node.py`
* **Debugging**

  * Use `rviz2` to inspect Nav2 (path, costmaps, TF).
  * Use `rqt` / `ros2 topic echo` to inspect YOLO detection topics.
  * Check Ollama and `vlm` node logs for model/VLM issues.

For any missing details, please open the packages inside this repository and check the `launch/`, `config/`, and `src/` folders.

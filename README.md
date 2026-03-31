# stereo-testing
This repo modifies stereo_image_proc built into ROS, performing an extra step of histogram stretch and CLAHE to attempt in improving number of valid points in SGM. This should only be used when running SGM, as SGBM would prefer images to be colored.
The added processing node is included in a Composable Node Container, ensuring shared memory to prevent costing too much bandwith.

## How to use
1. Clone this repo
2. Fix video path inside combined_publisher.py
3. colcon build && source install/setup.bash
4. Run with
```bash
ros2 launch stereo_publisher stereo_detection.launch.py
```

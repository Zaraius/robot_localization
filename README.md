Steps to run

ros2 launch robot_localization test_pf.py map_yaml:=src/robot_localization/maps/mac_1st_floor_final.yaml

ros2 bag play src/robot_localization/bags/macfirst_floor_take_2/macfirst_floor_take_2_0.db3 --clock

rviz2 -d ~/ros2_ws/src/robot_localization/rviz/turtlebot_bag_files.rviz

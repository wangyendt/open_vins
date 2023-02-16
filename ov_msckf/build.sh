# rm -rf build
mkdir build
cd build
cmake .. -DENABLE_ROS=OFF -DENABLE_ARUCO_TAGS=OFF -DBUILD_OV_EVAL=OFF -DDISABLE_MATPLOTLIB=ON
make -j12

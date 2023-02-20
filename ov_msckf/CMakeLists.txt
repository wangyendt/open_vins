cmake_minimum_required(VERSION 3.3)
project(ov_msckf)

set(BUILD_ROS 0)

if(BUILD_ROS)
    # Include libraries (if we don't have opencv 4, then fallback to opencv 3)
    # The OpenCV version needs to match the one used by cv_bridge otherwise you will get a segmentation fault!
    find_package(Eigen3 REQUIRED)
    find_package(OpenCV 4 QUIET)
    if (NOT OpenCV_FOUND)
        find_package(OpenCV 4 REQUIRED)
    endif ()
    find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time)
    find_package(Ceres REQUIRED)
    message(STATUS "OPENCV: " ${OpenCV_VERSION} " | BOOST: " ${Boost_VERSION} " | CERES: " ${Ceres_VERSION})

    # If we will compile with aruco support
    option(ENABLE_ARUCO_TAGS "Enable or disable aruco tag (disable if no contrib modules)" ON)
    if (NOT ENABLE_ARUCO_TAGS)
        add_definitions(-DENABLE_ARUCO_TAGS=0)
        message(WARNING "DISABLING ARUCOTAG TRACKING!")
    else ()
        add_definitions(-DENABLE_ARUCO_TAGS=1)
    endif ()

    # We need c++14 for ROS2, thus just require it for everybody
    # NOTE: To future self, hope this isn't an issue...
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)

    # Enable compile optimizations
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

    # Enable debug flags (use if you want to debug in gdb)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wuninitialized -fno-omit-frame-pointer")

    # Find our ROS version!
    # NOTE: Default to using the ROS1 package if both are in our enviroment
    # NOTE: https://github.com/romainreignier/share_ros1_ros2_lib_demo
    find_package(catkin QUIET COMPONENTS roscpp)
    find_package(ament_cmake QUIET)
    if (catkin_FOUND)
        message(STATUS "ROS *1* version found, building ROS1.cmake")
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROS1.cmake)
    elseif (ament_cmake_FOUND)
        message(STATUS "ROS *2* version found, building ROS2.cmake")
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROS2.cmake)
    else ()
        message(STATUS "No ROS versions found, building ROS1.cmake")
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROS1.cmake)
    endif ()
else()
    # Include libraries (if we don't have opencv 4, then fallback to opencv 3)
    # The OpenCV version needs to match the one used by cv_bridge otherwise you will get a segmentation fault!
    find_package(Eigen3 REQUIRED)
    find_package(OpenCV 4 QUIET)
    find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time)
    find_package(Ceres REQUIRED)
    message(STATUS "OPENCV: " ${OpenCV_VERSION} " | BOOST: " ${Boost_VERSION} " | CERES: " ${Ceres_VERSION})

    add_definitions(-DENABLE_ARUCO_TAGS=0)
    message(WARNING "DISABLING ARUCOTAG TRACKING!")

    # We need c++14 for ROS2, thus just require it for everybody
    # NOTE: To future self, hope this isn't an issue...
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)

    # Enable compile optimizations
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

    # Enable debug flags (use if you want to debug in gdb)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wuninitialized -fno-omit-frame-pointer")





    # include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROS1.cmake)
    add_definitions(-DROS_AVAILABLE=0)
    message(WARNING "BUILDING WITHOUT ROS!")
    include(GNUInstallDirs)
    set(CATKIN_PACKAGE_LIB_DESTINATION "${CMAKE_INSTALL_LIBDIR}")
    set(CATKIN_PACKAGE_BIN_DESTINATION "${CMAKE_INSTALL_BINDIR}")
    set(CATKIN_GLOBAL_INCLUDE_DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

    # Include our header files
    include_directories(
            src
            ${EIGEN3_INCLUDE_DIR}
            ${Boost_INCLUDE_DIRS}
            ${CERES_INCLUDE_DIRS}
            ${catkin_INCLUDE_DIRS}
    )

    # Set link libraries used by all binaries
    list(APPEND thirdparty_libraries
            ${Boost_LIBRARIES}
            ${OpenCV_LIBRARIES}
            ${CERES_LIBRARIES}
            ${catkin_LIBRARIES}
    )


    # If we are not building with ROS then we need to manually link to its headers
    # This isn't that elegant of a way, but this at least allows for building without ROS
    # If we had a root cmake we could do this: https://stackoverflow.com/a/11217008/7718197
    # But since we don't we need to basically build all the cpp / h files explicitly :(

    message(STATUS "MANUALLY LINKING TO OV_CORE LIBRARY....")
    include_directories(${CMAKE_SOURCE_DIR}/../ov_core/src/)
    file(GLOB_RECURSE OVCORE_LIBRARY_SOURCES "${CMAKE_SOURCE_DIR}/../ov_core/src/*.cpp")
    list(FILTER OVCORE_LIBRARY_SOURCES EXCLUDE REGEX ".*test_webcam\\.cpp$")
    list(FILTER OVCORE_LIBRARY_SOURCES EXCLUDE REGEX ".*test_tracking\\.cpp$")
    list(APPEND LIBRARY_SOURCES ${OVCORE_LIBRARY_SOURCES})
    file(GLOB_RECURSE OVCORE_LIBRARY_HEADERS "${CMAKE_SOURCE_DIR}/../ov_core/src/*.h")
    list(APPEND LIBRARY_HEADERS ${OVCORE_LIBRARY_HEADERS})

    message(STATUS "MANUALLY LINKING TO OV_INIT LIBRARY....")
    include_directories(${CMAKE_SOURCE_DIR}/../ov_init/src/)
    file(GLOB_RECURSE OVINIT_LIBRARY_SOURCES "${CMAKE_SOURCE_DIR}/../ov_init/src/*.cpp")
    list(FILTER OVINIT_LIBRARY_SOURCES EXCLUDE REGEX ".*test_dynamic_init\\.cpp$")
    list(FILTER OVINIT_LIBRARY_SOURCES EXCLUDE REGEX ".*test_dynamic_mle\\.cpp$")
    list(FILTER OVINIT_LIBRARY_SOURCES EXCLUDE REGEX ".*test_simulation\\.cpp$")
    list(FILTER OVINIT_LIBRARY_SOURCES EXCLUDE REGEX ".*Simulator\\.cpp$")
    list(APPEND LIBRARY_SOURCES ${OVINIT_LIBRARY_SOURCES})
    file(GLOB_RECURSE OVINIT_LIBRARY_HEADERS "${CMAKE_SOURCE_DIR}/../ov_init/src/*.h")
    list(FILTER OVINIT_LIBRARY_HEADERS EXCLUDE REGEX ".*Simulator\\.h$")
    list(APPEND LIBRARY_HEADERS ${OVINIT_LIBRARY_HEADERS})


    ##################################################
    # Make the shared library
    ##################################################

    list(APPEND LIBRARY_SOURCES
            src/dummy.cpp
            src/sim/Simulator.cpp
            src/state/State.cpp
            src/state/StateHelper.cpp
            src/state/Propagator.cpp
            src/core/VioManager.cpp
            src/core/VioManagerHelper.cpp
            src/update/UpdaterHelper.cpp
            src/update/UpdaterMSCKF.cpp
            src/update/UpdaterSLAM.cpp
            src/update/UpdaterZeroVelocity.cpp
    )


    ##################################################
    file(GLOB_RECURSE LIBRARY_HEADERS "src/*.h")
    add_library(ov_msckf_lib SHARED ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
    target_link_libraries(ov_msckf_lib ${thirdparty_libraries})
    target_include_directories(ov_msckf_lib PUBLIC src/)


    ##################################################
    # Make binary files!
    ##################################################
    add_executable(run_simulation src/run_simulation.cpp)
    target_link_libraries(run_simulation ov_msckf_lib ${thirdparty_libraries})
    install(TARGETS run_simulation
            ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
            LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
            RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )

    add_executable(run_custom_dataset src/run_custom_dataset_without_ros.cpp)
    target_link_libraries(run_custom_dataset ov_msckf_lib ${thirdparty_libraries})
    install(TARGETS run_simulation
            ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
            LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
            RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )



    # Find our ROS version!
    # NOTE: Default to using the ROS1 package if both are in our enviroment
    # NOTE: https://github.com/romainreignier/share_ros1_ros2_lib_demo
    # find_package(catkin QUIET COMPONENTS roscpp)
    # find_package(ament_cmake QUIET)
    # if (catkin_FOUND)
    #     message(STATUS "ROS *1* version found, building ROS1.cmake")
    #     include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROS1.cmake)
    # elseif (ament_cmake_FOUND)
    #     message(STATUS "ROS *2* version found, building ROS2.cmake")
    #     include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROS2.cmake)
    # else ()
    #     message(STATUS "No ROS versions found, building ROS1.cmake")
    #     include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROS1.cmake)
    # endif ()
endif()


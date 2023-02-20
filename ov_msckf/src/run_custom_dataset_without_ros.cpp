//
// Created by wayne on 2023/2/16.
//


#include <memory>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "state/State.h"

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "utils/dataset_reader.h"
#include "utils/sensor_data.h"
#include "utils/ConfigReader.h"

using namespace ov_msckf;

std::shared_ptr<VioManager> sys;

void ReadImu(std::vector<std::pair<double, std::array<double, 6>>> &imu_data_vec, const std::string &imu_file_path) {
    ifstream ifs;
    ifs.open(imu_file_path.c_str());
    imu_data_vec.reserve(5000);

    while (!ifs.eof()) {
        string line;
        getline(ifs, line);
        if (line[0] == '#')
            continue;

        if (!line.empty()) {
            for (auto &a: line) {
                if (a == ',')
                    a = ' ';
            }

            std::pair<double, std::array<double, 6>> data;

            std::stringstream ss;
            ss << line;
            ss >> data.first >> data.second[0] >> data.second[1] >> data.second[2] >>
               data.second[3] >> data.second[4] >> data.second[5];
            data.first /= 1e9;

            imu_data_vec.push_back(data);
        }
    }
}

void ReadImageNames(std::vector<std::pair<double, std::string>> &img_names_vec, const std::string &timestamp_file_path) {
    ifstream ifs;
    ifs.open(timestamp_file_path.c_str());
    img_names_vec.reserve(500);

    while (!ifs.eof()) {
        string line;
        getline(ifs, line);
        if (line[0] == '#')
            continue;

        if (!line.empty()) {
            for (auto &a: line) {
                if (a == ',')
                    a = ' ';
            }

            std::pair<double, std::string> data;

            std::stringstream ss;
            ss << line;
            ss >> data.first >> data.second;
            data.first /= 1e9;

            img_names_vec.push_back(data);
        }
    }
}


void ReadGTFile(std::map<double, Eigen::Matrix<double, 17, 1>> &gt_states, const std::string &gt_file_path) {
    // Clear any old data
    gt_states.clear();

    ifstream ifs;
    ifs.open(gt_file_path.c_str());

    while (!ifs.eof()) {
        string line;
        getline(ifs, line);
        if (line[0] == '#')
            continue;

        if (!line.empty()) {
            for (auto &a: line) {
                if (a == ',')
                    a = ' ';
            }

            Eigen::Matrix<double, 17, 1> data = Eigen::Matrix<double, 17, 1>::Zero();

            std::stringstream ss;
            ss << line;
            for (size_t i = 0; i < 17; i++) {
                ss >> data[i];
            }
            gt_states.insert({1e-9 * data(0, 0), data});
        }
    }
}

bool get_gt_state(double timestep, Eigen::Matrix<double, 17, 1> &imustate, std::map<double, Eigen::Matrix<double, 17, 1>> &gt_states) {

    // Check that we even have groundtruth loaded
    if (gt_states.empty()) {
        PRINT_ERROR(RED "Groundtruth data loaded is empty, make sure you call load before asking for a state.\n" RESET);
        return false;
    }

    // Loop through gt states and find the closest time stamp
    double closest_time = std::numeric_limits<double>::max();
    auto it0 = gt_states.begin();
    while (it0 != gt_states.end()) {
        if (std::abs(it0->first - timestep) < std::abs(closest_time - timestep)) {
            closest_time = it0->first;
        }
        it0++;
    }

    // If close to this timestamp, then use it
    if (std::abs(closest_time - timestep) < 0.005) {
        //printf("init DT = %.4f\n", std::abs(closest_time-timestep));
        //printf("timestamp = %.15f\n", closest_time);
        timestep = closest_time;
    }

    // Check that we have the timestamp in our GT file
    if (gt_states.find(timestep) == gt_states.end()) {
        PRINT_ERROR(YELLOW "Unable to find %.6f timestamp in GT file, wrong GT file loaded???\n" RESET, timestep);
        return false;
    }

    // Get the GT state vector
    Eigen::Matrix<double, 17, 1> state = gt_states[timestep];

    // Our "fixed" state vector from the ETH GT format [q,p,v,bg,ba]
    imustate(0, 0) = timestep; //time
    imustate(1, 0) = state(5, 0); //quat
    imustate(2, 0) = state(6, 0);
    imustate(3, 0) = state(7, 0);
    imustate(4, 0) = state(4, 0);
    imustate(5, 0) = state(1, 0); //pos
    imustate(6, 0) = state(2, 0);
    imustate(7, 0) = state(3, 0);
    imustate(8, 0) = state(8, 0); //vel
    imustate(9, 0) = state(9, 0);
    imustate(10, 0) = state(10, 0);
    imustate(11, 0) = state(11, 0); //bg
    imustate(12, 0) = state(12, 0);
    imustate(13, 0) = state(13, 0);
    imustate(14, 0) = state(14, 0); //ba
    imustate(15, 0) = state(15, 0);
    imustate(16, 0) = state(16, 0);

    // Success!
    return true;
}

// Main function
int main(int argc, char **argv) {

    // 设置调试等级
//    ov_core::Printer::setPrintLevel(std::string("DEBUG"));

    if (argc != 2) {
        std::cout << "Usage: ./exe path_to_config_file" << std::endl;
        return -1;
    }

    std::string config_file_path(argv[1]);

    std::string euroc_dataset_path = "";
    std::string gt_file_path = "";
    std::string output_pose_file_path = "pose.txt";
    double t_start = 0, t_end = -1;

    cv::FileStorage fs;
    fs.open(config_file_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error! cannot open config file: " << config_file_path << std::endl;
        return -1;
    }
    ReadParamCVFs(euroc_dataset_path, fs, "dataset_path");
    ReadParamCVFs(gt_file_path, fs, "gt_file_path");
    ReadParamCVFs(output_pose_file_path, fs, "output_pose_file_path");
    ReadParamCVFs(t_start, fs, "t_start");
    ReadParamCVFs(t_end, fs, "t_end");
    fs.release();

    std::string imu_file_path = euroc_dataset_path + "/mav0/imu0/data.csv";
    std::string timestamp_file_path = euroc_dataset_path + "/mav0/cam0/data.csv";
    std::string imgs_folder = euroc_dataset_path + "/mav0/cam0/data";

    // Create our VIO system
    VioManagerOptions params;
    ReadConfig(params, config_file_path);

    sys = std::make_shared<VioManager>(params);

    std::ofstream ofs_pose_cam0;
    ofs_pose_cam0.open(output_pose_file_path);
    if (!ofs_pose_cam0.is_open()) {
        std::cout << "Error! cannot open file " << output_pose_file_path << std::endl;
        return 1;
    }

    std::ofstream ofs_pose_imu;
    ofs_pose_imu.open(output_pose_file_path + ".imu");
    if (!ofs_pose_imu.is_open()) {
        std::cout << "Error! cannot open file " << output_pose_file_path + ".imu" << std::endl;
        return 1;
    }

    // --------------  Read IMU data --------------------
    std::vector<std::pair<double, std::array<double, 6>>> imu_data_vec;
    ReadImu(imu_data_vec, imu_file_path);
    PRINT_INFO(GREEN "imu_data_size: %ld\n" RESET, imu_data_vec.size());

    // -------------- Read camera timestamps ------------
    std::vector<std::pair<double, std::string>> img_names_vec;
    ReadImageNames(img_names_vec, timestamp_file_path);
    PRINT_INFO(GREEN "image_data_size: %ld\n" RESET, img_names_vec.size());

    // -------------- Read ground truth -----------------
    std::map<double, Eigen::Matrix<double, 17, 1>> gt_states;
    if (!gt_file_path.empty()) {
        ReadGTFile(gt_states, gt_file_path);
    }





    std::deque<ov_core::CameraData> camera_queue;
    std::mutex camera_queue_mtx;
    std::atomic<bool> thread_update_running(false);
    std::map<int, double> camera_last_timestamp;
    auto cam_id0 = 0;

    std::map<double, std::pair<std::string, int>> sensor_msgs;
    for (auto i = 0; i < imu_data_vec.size(); ++i) {
        sensor_msgs[imu_data_vec[i].first] = {"IMU", i};
    }
    for (auto i = 0; i < img_names_vec.size(); ++i) {
        sensor_msgs[img_names_vec[i].first] = {"Cam", i};
    }
    for (auto iter = sensor_msgs.begin(); iter != sensor_msgs.end(); ++iter) {
        auto timestamp = (*iter).first;
        auto sensor_type = (*iter).second.first;
        auto sensor_idx = (*iter).second.second;
        if (sensor_type == "IMU") {

            // convert into correct format
            ov_core::ImuData message;
            message.timestamp = timestamp;
            message.wm << imu_data_vec[sensor_idx].second[0], imu_data_vec[sensor_idx].second[1], imu_data_vec[sensor_idx].second[2];
            message.am << imu_data_vec[sensor_idx].second[3], imu_data_vec[sensor_idx].second[4], imu_data_vec[sensor_idx].second[5];

//    PRINT_INFO(YELLOW "ax=%f,ay=%f,az=%f,gx=%f,gy=%f,gz=%f\n" RESET, msg->linear_acceleration.x,msg->linear_acceleration.y,msg->linear_acceleration.z,msg->angular_velocity.x,msg->angular_velocity.y,msg->angular_velocity.z);
//            PRINT_INFO(YELLOW "[IMU]timestamp=%f\n" RESET, message.timestamp);

            // send it to our VIO system
            sys->feed_measurement_imu(message);
//            visualize_odometry(message.timestamp);

            // If the processing queue is currently active / running just return so we can keep getting measurements
            // Otherwise create a second thread to do our update in an async manor
            // The visualization of the state, images, and features will be synchronous with the update!
            if (thread_update_running)
                continue;
            thread_update_running = true;
            std::thread thread([&] {
                // Lock on the queue (prevents new images from appending)
                std::lock_guard<std::mutex> lck(camera_queue_mtx);

                // Count how many unique image streams
                std::map<int, bool> unique_cam_ids;
                for (const auto &cam_msg : camera_queue) {
                    unique_cam_ids[cam_msg.sensor_ids.at(0)] = true;
                }

                // If we do not have enough unique cameras then we need to wait
                // We should wait till we have one of each camera to ensure we propagate in the correct order
                auto params = sys->get_params();
                size_t num_unique_cameras = (params.state_options.num_cameras == 2) ? 1 : params.state_options.num_cameras;
                if (unique_cam_ids.size() == num_unique_cameras) {

                    // Loop through our queue and see if we are able to process any of our camera measurements
                    // We are able to process if we have at least one IMU measurement greater than the camera time
                    double timestamp_imu_inC = message.timestamp - sys->get_state()->_calib_dt_CAMtoIMU->value()(0);
                    while (!camera_queue.empty() && camera_queue.at(0).timestamp < timestamp_imu_inC) {
                        auto rT0_1 = boost::posix_time::microsec_clock::local_time();
                        double update_dt = 100.0 * (timestamp_imu_inC - camera_queue.at(0).timestamp);
                        PRINT_INFO(YELLOW "img ts=%f,imu ts=%f\n" RESET,camera_queue.at(0).timestamp,message.timestamp);
                        sys->feed_measurement_camera(camera_queue.at(0));
//                        visualize();
                        camera_queue.pop_front();
                        auto rT0_2 = boost::posix_time::microsec_clock::local_time();
                        double time_total = (rT0_2 - rT0_1).total_microseconds() * 1e-6;
                        PRINT_INFO(BLUE "[TIME]: %.4f seconds total (%.1f hz, %.2f ms behind)\n" RESET, time_total, 1.0 / time_total, update_dt);
                    }
                }
                thread_update_running = false;
            });

            // If we are single threaded, then run single threaded
            // Otherwise detach this thread so it runs in the background!
            if (!sys->get_params().use_multi_threading_subs) {
                thread.join();
            } else {
                thread.detach();
            }
        } else if (sensor_type == "Cam") {

            // Check if we should drop this image
            double time_delta = 1.0 / sys->get_params().track_frequency;
            if (camera_last_timestamp.find(cam_id0) != camera_last_timestamp.end() && timestamp < camera_last_timestamp.at(cam_id0) + time_delta) {
                continue;
            }
            camera_last_timestamp[cam_id0] = timestamp;

            // Get the image
//            cv_bridge::CvImageConstPtr cv_ptr;
//            try {
//                cv_ptr = cv_bridge::toCvShare(msg0, sensor_msgs::image_encodings::MONO8);
//            } catch (cv_bridge::Exception &e) {
//                PRINT_ERROR("cv_bridge exception: %s", e.what());
//                return;
//            }
            cv::Mat img_left = cv::imread(euroc_dataset_path + "/mav0/cam0/data/" + img_names_vec[sensor_idx].second, cv::IMREAD_GRAYSCALE);

            // Create the measurement
            ov_core::CameraData message;
            message.timestamp = timestamp;
            message.sensor_ids.push_back(cam_id0);
            message.images.push_back(img_left.clone());

            // Load the mask if we are using it, else it is empty
            // TODO: in the future we should get this from external pixel segmentation
            if (sys->get_params().use_mask) {
                message.masks.push_back(sys->get_params().masks.at(cam_id0));
            } else {
//                PRINT_INFO(YELLOW "[IMAGE]timestamp=%f,timestamp_raw=%f,sensor_id=%ld\n" RESET, message.timestamp, timestamp, cam_id0);
                message.masks.push_back(cv::Mat::zeros(img_left.rows, img_left.cols, CV_8UC1));
            }

            // append it to our queue of images
            std::lock_guard<std::mutex> lck(camera_queue_mtx);
            camera_queue.push_back(message);
            std::sort(camera_queue.begin(), camera_queue.end());


            // display tracking result
            cv::Mat tracked_img = sys->get_historical_viz_image();
            if (!tracked_img.empty()) {
                cv::imshow("tracked_img", tracked_img);
            }
            int key = cv::waitKey(5);
            if (key == 'q' || key == 'Q') {
                break;
            }

            if (sys->initialized()) {

                PRINT_INFO(WHITE "Start to save stuff!" RESET);

                // Get current state
                auto state = sys->get_state();


                // save pose of cam0
                {
                    double t_cam = state->_timestamp;

                    auto R_iw = state->_imu->Rot();
                    auto t_wi = state->_imu->pos();

                    auto R_ci = state->_calib_IMUtoCAM.at(0)->Rot();
                    auto t_ci = state->_calib_IMUtoCAM.at(0)->pos();

                    Eigen::Matrix3d R_ic = R_ci.transpose();
                    Eigen::Vector3d t_ic = -R_ci.transpose() * t_ci;

                    auto R_wi = R_iw.transpose();
                    auto R_wc = R_wi * R_ic;
                    auto t_wc = R_wi * t_ic + t_wi;

                    Eigen::Quaterniond q_wc(R_wc);

                    ofs_pose_cam0 << std::setprecision(18)
                                  << t_cam << " "
                                  << t_wc[0] << " " << t_wc[1] << " " << t_wc[2] << " "
                                  << q_wc.x() << " " << q_wc.y() << " " << q_wc.z() << " " << q_wc.w()
                                  << std::endl;
                    ofs_pose_cam0.flush();
                }

                // save pose of imu
                {
                    double timestamp = state->_timestamp;
                    auto pos = state->_imu->pos();
                    auto ori = state->_imu->quat(); // R_iw

                    ofs_pose_imu << std::setprecision(18)
                                 << timestamp << " "
                                 << pos[0] << " " << pos[1] << " " << pos[2] << " "
                                 << ori[0] << " " << ori[1] << " " << ori[2] << " " << ori[3]
                                 << std::endl;
                    ofs_pose_imu.flush();
                }
            }
        }
//        std::cout << setprecision(20) << (*iter).first << ":" << (*iter).second.first << "--" << (*iter).second.second << std::endl;
    }

    return EXIT_SUCCESS;

//    size_t num_frame = img_names_vec.size();
//    size_t idx_start = 0, idx_end = num_frame;
//    if (t_start >= 0) {
//        size_t idx = 0;
//        for (; idx < num_frame; idx++) {
//            if (img_names_vec[idx].first - img_names_vec[0].first >= t_start) {
//                break;
//            }
//        }
//        idx_start = idx;
//    }
//
//    if (t_end > 0) {
//        size_t idx = idx_start;
//        for (; idx < num_frame; idx++) {
//            if (img_names_vec[idx].first - img_names_vec[0].first >= t_end) {
//                break;
//            }
//        }
//        idx_end = idx;
//    }
//
//    PRINT_INFO(WHITE "idx start = %ld, end = %ld\n" RESET, idx_start, idx_end);
//
//    auto iter_imu_data = imu_data_vec.begin();
//    for (size_t idx_frame = idx_start; idx_frame < idx_end; idx_frame++) {
//        double t_img = img_names_vec[idx_frame].first;
//
//        std::vector<ov_core::ImuData> measure_imu_vec;
//        while (iter_imu_data != imu_data_vec.end() && iter_imu_data->first < t_img + sys->get_state()->_calib_dt_CAMtoIMU->value()(0)) {
//            auto imu_data = iter_imu_data->second;
//            ov_core::ImuData measure_imu;
//            measure_imu.timestamp = iter_imu_data->first;
//            measure_imu.am << imu_data[3], imu_data[4], imu_data[5];
//            measure_imu.wm << imu_data[0], imu_data[1], imu_data[2];
//
////            PRINT_INFO(WHITE "ax=%f,ay=%f,az=%f,gx=%f,gy=%f,gz=%f\n" RESET, imu_data[3],imu_data[4],imu_data[5],imu_data[0],imu_data[1],imu_data[2]);
//            PRINT_INFO(WHITE "imuts=%f,imgts=%f\n" RESET, iter_imu_data->first, img_names_vec[idx_frame].first);
//
//            measure_imu_vec.push_back(measure_imu);
//            iter_imu_data++;
//        }
//
//        PRINT_INFO(WHITE "imu size=%ld, at frame %ld\n", measure_imu_vec.size(), idx_frame);
//
//        for (const auto &measure_imu: measure_imu_vec) {
//            sys->feed_measurement_imu(measure_imu);
//        }
//
//        cv::Mat img_left, img_right;
//        img_left = cv::imread(euroc_dataset_path + "/mav0/cam0/data/" + img_names_vec[idx_frame].second, cv::IMREAD_GRAYSCALE);
//        if (img_left.empty()) {
//            std::cout << "img at " << t_img << " is empty!" << std::endl;
//            continue;
//        }
//        if (img_left.channels() == 3) {
//            cv::cvtColor(img_left, img_left, cv::COLOR_BGR2GRAY);
//        }
//
//        if (params.state_options.num_cameras == 2) {
//            img_right = cv::imread(euroc_dataset_path + "/mav0/cam1/data/" + img_names_vec[idx_frame].second, cv::IMREAD_GRAYSCALE);
//            if (img_right.empty()) {
//                std::cout << "img2 at " << t_img << " is empty!" << std::endl;
//                continue;
//            }
//            if (img_right.channels() == 3) {
//                cv::cvtColor(img_right, img_right, cv::COLOR_BGR2GRAY);
//            }
//        }
//
//        Eigen::Matrix<double, 17, 1> imustate;
//        if (!gt_states.empty() && !sys->initialized() && get_gt_state(t_img, imustate, gt_states)) {
//            //biases are pretty bad normally, so zero them
//            //imustate.block(11,0,6,1).setZero();
//            sys->initialize_with_gt(imustate);
//        } else if (gt_states.empty() || sys->initialized()) {
//            if (params.state_options.num_cameras == 1) {
//                ov_core::CameraData measure_img;
//                measure_img.timestamp = t_img;
//                measure_img.sensor_ids.push_back(0);
//                measure_img.images.push_back(img_left);
//                measure_img.masks.push_back(cv::Mat::zeros(img_left.size(), img_left.type()));
//                sys->feed_measurement_camera(measure_img);
//            } else if (params.state_options.num_cameras == 2 && params.use_stereo) {
////                ov_core::CameraData measure_img;
////                measure_img.timestamp = t_img;
////                measure_img.sensor_ids.push_back(0);
////                measure_img.images.push_back(img_left);
////                measure_img.sensor_ids.push_back(1);
////                measure_img.images.push_back(img_right);
////                sys->feed_measurement_camera(measure_img);
//            } else {
////                ov_core::CameraData measure_img_left;
////                measure_img_left.timestamp = t_img;
////                measure_img_left.sensor_ids.push_back(0);
////                measure_img_left.images.push_back(img_left);
////                sys->feed_measurement_camera(measure_img_left);
////
////                ov_core::CameraData measure_img_right;
////                measure_img_right.timestamp = t_img;
////                measure_img_right.sensor_ids.push_back(1);
////                measure_img_right.images.push_back(img_right);
////                sys->feed_measurement_camera(measure_img_right);
//            }
//        }
//
//        // display tracking result
//        cv::Mat tracked_img = sys->get_historical_viz_image();
//        if (!tracked_img.empty()) {
//            cv::imshow("tracked_img", tracked_img);
//        }
//        int key = cv::waitKey(5);
//        if (key == 'q' || key == 'Q') {
//            break;
//        }
//
//        if (sys->initialized()) {
//
//            PRINT_INFO(WHITE "Start to save stuff!" RESET);
//
//            // Get current state
//            auto state = sys->get_state();
//
//
//            // save pose of cam0
//            {
//                double t_cam = state->_timestamp;
//
//                auto R_iw = state->_imu->Rot();
//                auto t_wi = state->_imu->pos();
//
//                auto R_ci = state->_calib_IMUtoCAM.at(0)->Rot();
//                auto t_ci = state->_calib_IMUtoCAM.at(0)->pos();
//
//                Eigen::Matrix3d R_ic = R_ci.transpose();
//                Eigen::Vector3d t_ic = -R_ci.transpose() * t_ci;
//
//                auto R_wi = R_iw.transpose();
//                auto R_wc = R_wi * R_ic;
//                auto t_wc = R_wi * t_ic + t_wi;
//
//                Eigen::Quaterniond q_wc(R_wc);
//
//                ofs_pose_cam0 << std::setprecision(18)
//                              << t_cam << " "
//                              << t_wc[0] << " " << t_wc[1] << " " << t_wc[2] << " "
//                              << q_wc.x() << " " << q_wc.y() << " " << q_wc.z() << " " << q_wc.w()
//                              << std::endl;
//                ofs_pose_cam0.flush();
//            }
//
//            // save pose of imu
//            {
//                double timestamp = state->_timestamp;
//                auto pos = state->_imu->pos();
//                auto ori = state->_imu->quat(); // R_iw
//
//                ofs_pose_imu << std::setprecision(18)
//                             << timestamp << " "
//                             << pos[0] << " " << pos[1] << " " << pos[2] << " "
//                             << ori[0] << " " << ori[1] << " " << ori[2] << " " << ori[3]
//                             << std::endl;
//                ofs_pose_imu.flush();
//            }
//        }
//    }
//
//    ofs_pose_cam0.flush();
//    ofs_pose_cam0.close();
//    ofs_pose_imu.flush();
//    ofs_pose_imu.close();
//
//    // Done!
//    return EXIT_SUCCESS;
}



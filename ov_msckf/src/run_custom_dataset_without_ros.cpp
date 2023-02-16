// #include <csignal>
#include <memory>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
// #include "sim/Simulator.h"
// #include "utils/colors.h"
#include "utils/dataset_reader.h"
// #include "utils/print.h"
#include "utils/sensor_data.h"

using namespace ov_msckf;


int main(int argc, char **argv){
    // 设置调试等级
    ov_core::Printer::setPrintLevel(std::string("DEBUG"));

    // 设置配置路径
    std::string config_path = argc > 1 ? argv[1] : "undefined";

    // 定义parser和params读取配置信息
    VioManagerOptions params;
    auto parser = std::make_shared<ov_core::YamlParser>(config_path);
    params.print_and_load(parser);

    // 定义核心VioManager
    auto sys = std::make_shared<VioManager>(params);
    return EXIT_SUCCESS;
}


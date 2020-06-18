#include "detector_types.hpp"
#include "utils/json/json.h"
#include "utils/json/json-forwards.h"
#include "utils/logging.hpp"

long DetectorParams::Read(const string& config_file){
    using namespace Json;
    using namespace easycv;
    Reader reader;
    Value root;
    std::ifstream ifs(config_file);
    bool ret = reader.parse(ifs, root);
    if (!ret) {
        LOG(INFO) << "can not parse " << config_file;
        return 0L;
    }
    model_path = root["model_path"].asString();
    model_type = root["model_type"].asString();
    video_detect_every_n = root["video_detect_every_n"].asInt();
    // string type_str = root["model_type"].asString();
    // if(type_str.compare("SSD")==0)
    // {
    //     type = DetectorType::SSD;
    // }
    // else if(type_str.compare("FasterRcnn")==0)
    // {
    //     type = DetectorType::FasterRcnn;
    // }
    // else if(type_str.compare("MaskRcnn")==0)
    // {
    //     type = DetectorType::MaskRcnn;
    // }
    // else{
    //     LOG(ERROR)<<"Not supported model "<<type_str;
    //     return -1L;
    // }

    device_id = root["device_id"].asInt();
    return 1L;

}

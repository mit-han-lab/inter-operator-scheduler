//
// Created by yaoyao on 8/24/19.
//

#ifndef MUSIC_UTILS_H
#define MUSIC_UTILS_H

#include "dist/json/json.h"

#define DLL __attribute__((visibility("default")))

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(f) do {                                             \
    auto status = f;                                                   \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      std::stringstream _error;                                        \
      _error << "CUDNN failure: " << (status != CUDNN_STATUS_SUCCESS) << " " << status << " " << cudnnGetErrorString(status); \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCUDA(f) do {                                              \
    auto status = f;                                                   \
    if (status != 0) {                                                 \
      std::stringstream _error;                                        \
      _error << "Cuda failure: " << status << " " << cudaGetErrorString(status); \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

Json::Value json_from_cstr(const char *str);
bool file_exists(const char *str);

#endif //MUSIC_UTILS_H

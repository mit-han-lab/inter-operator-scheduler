//
// Created by yyding on 8/25/19.
//
#include <fstream>
#include <sstream>
#include "dist/json/json.h"

Json::Value json_from_cstr(const char *str) {
    Json::Value value;
    std::stringstream ss(str);
    ss >> value;
    return value;
}

bool file_exists(const char *str) {
    std::ifstream f(str);
    return f.good();
}

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <malloc.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include "NvInfer.h"
#include "utils/utils.h"


#define MAX_NUM_NODES 500

using namespace nvinfer1;

const DataType dtype = DataType::kFLOAT;

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
//        if (severity != Severity::kINFO)
//            std::cout << msg << std::endl;
    }
} gLogger;

struct Profiler : public IProfiler
{
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;
    static const int TIMING_ITERATIONS = 10;

    virtual void reportLayerTime(const char* layerName, float ms)
    {
        auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
        if (record == mProfile.end())
            mProfile.push_back(std::make_pair(layerName, ms));
        else
            record->second += ms;
    }

    void printLayerTimes()
    {
        float totalTime = 0;
        for (size_t i = 0; i < mProfile.size(); i++)
        {
            printf("%s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / TIMING_ITERATIONS); // %-400.400s
            totalTime += mProfile[i].second;
        }
        printf("Time over all layers: %4.3f\n", totalTime / TIMING_ITERATIONS);
    }
} gProfiler;

struct WeightManager {
    float* weights[MAX_NUM_NODES][4];
    float* empty[4];
    std::map<std::string,int> name2index;
    void init(int num_convs, char **convs_name, float **convs_weight, float **convs_bias) {
        int index = 0;
        name2index.clear();
        for(int i = 0; i < num_convs; i++) {
            weights[index][0] = convs_weight[i];
            weights[index][1] = convs_bias[i];
            weights[index][2] = weights[index][3] = nullptr;
            name2index[convs_name[i]] = index++;
        }
        for(int i = 0; i < 4; i++)
            empty[i] = nullptr;
    }
    float** get_weights(std::string name) {
        if(name2index.count(name))
            return weights[name2index[name]];
        else {
            return empty;
        }
    }
}weightManager;

void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    checkCUDA(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

ITensor * get_input(INetworkDefinition *network, Json::Value input_config, std::map<std::string, ITensor*> &node2tensor, Json::Value node_config) {
    int num_terms = (int)input_config.size();
    std::vector<ITensor*> terms;
    for(int i = 0; i < num_terms; i++) {
        Json::Value term_config = input_config[i];
        int num_values = term_config.size();
        std::vector<ITensor*> values;
        for(int j = 0; j < num_values; j++) {
            Json::Value value_config = term_config[j];
            std::string name = value_config[0].asString();
            int begin = value_config[1].asInt();
            int end = value_config[2].asInt();
            ITensor *tensor = node2tensor[name];
            if(begin != 0 || end != tensor->getDimensions().d[0]) {
                Dims shape = tensor->getDimensions();
                Dims3 start(begin, 0, 0);
                Dims3 size(end - begin, shape.d[1], shape.d[2]);
                Dims3 stride(1, 1, 1);
                tensor = network->addSlice(*tensor, start, size, stride)->getOutput(0);
            }
            values.push_back(tensor);
        }
        ITensor * sum = values[0];
        for(size_t j = 1; j < values.size(); j++) {
            if(node_config["type"].asString() == "element") {
                std::string op_type = node_config["op_type"].asString();
                if(op_type == "mul") {
                    sum = network->addElementWise(*sum, *values[j], ElementWiseOperation::kPROD)->getOutput(0);
                } else if(op_type == "add") {
                    sum = network->addElementWise(*sum, *values[j], ElementWiseOperation::kSUM)->getOutput(0);
                } else {
                    FatalError("not supported op_type");
                }
            } else {
                sum = network->addElementWise(*sum, *values[j], ElementWiseOperation::kSUM)->getOutput(0);
            }
        }
        terms.push_back(sum);
    }
    if(terms.size() == 1)
        return terms[0];
    else {
        return network->addConcatenation(terms.data(), terms.size())->getOutput(0);
    }
}

void placeholder(INetworkDefinition *network, Json::Value config, std::map<std::string, ITensor*> &node2tensor) {
    Json::Value shape = config["output_shape"];
    std::string name = config["name"].asString();
    int c = shape[0].asInt();
    int h = shape[1].asInt();
    int w = shape[2].asInt();
    ITensor *tensor = network->addInput(config["name"].asString().c_str(), dtype, Dims3(c, h, w));
    node2tensor[name] = tensor;
}

void identity(INetworkDefinition *network, Json::Value config, std::map<std::string, ITensor*> &node2tensor) {
    ITensor *tensor = get_input(network, config["inputs"], node2tensor, config);
    std::string name = config["name"].asString();
    node2tensor[name] = tensor;
}

ITensor *conv2d_op(INetworkDefinition *network, ITensor *tensor, int out_channels, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w, int groups, float *filter, float *bias) {
    int in_channels = tensor->getDimensions().d[0];
    Weights filter_weights = {dtype, filter, (in_channels / groups) * out_channels * kernel_h * kernel_w};
    Weights bias_weights = {dtype, bias, out_channels};
    bool free_filter = false, free_bias = false;
    if(filter_weights.values == nullptr) {
        filter_weights.values = malloc(sizeof(float) * filter_weights.count);
        free_filter = true;
    }
    if(bias_weights.values == nullptr) {
        bias_weights.values = malloc(sizeof(float) * bias_weights.count);
        free_bias = true;
    }
    auto conv = network->addConvolution(*tensor, out_channels, DimsHW(kernel_h, kernel_w), filter_weights, bias_weights);
    if(free_filter)
        free((void*)filter_weights.values);
    if(free_bias)
        free((void*)bias_weights.values);

    conv->setStride(DimsHW(stride_h, stride_w));
    conv->setPadding(DimsHW(padding_h, padding_w));
    conv->setNbGroups(groups);
    return conv->getOutput(0);
}


void conv2d(INetworkDefinition *network, Json::Value config, std::map<std::string, ITensor*> &node2tensor) {
    ITensor *tensor = get_input(network, config["inputs"], node2tensor, config);
    std::string name = config["name"].asString();
    int out_channels = config["out_channels"].asInt();
    int kernel_h = config["kernel"][0].asInt();
    int kernel_w = config["kernel"][1].asInt();
    int stride_h = config["stride"][0].asInt();
    int stride_w = config["stride"][1].asInt();
    int padding_h = config["padding"][0].asInt();
    int padding_w = config["padding"][1].asInt();
    int groups = config["groups"].asInt();
    std::string act_str = config["act"].asString();
    float **weights = weightManager.get_weights(name);
    tensor = conv2d_op(network, tensor, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, groups, weights[0], weights[1]);
    if(act_str == "relu") {
        tensor = network->addActivation(*tensor, ActivationType::kRELU)->getOutput(0);
    } else if(act_str == "sigmoid") {
        tensor = network->addActivation(*tensor, ActivationType::kSIGMOID)->getOutput(0);
    } else if(act_str == "tanh") {
        tensor = network->addActivation(*tensor, ActivationType::kTANH)->getOutput(0);
    } else if(act_str == "identity") {
    } else {
        FatalError("");
    }
    node2tensor[name] = tensor;
}

void print(const char *name, float *data, int len) {
    fprintf(stderr, "%s %d:", name, len);
    for(int i = 0; i < len; i++)
        fprintf(stderr, "%.3f ", data[i]);
    fprintf(stderr, "\n");
}

void pool2d(INetworkDefinition *network, Json::Value config, std::map<std::string, ITensor*> &node2tensor) {
    ITensor *tensor = get_input(network, config["inputs"], node2tensor, config);
    std::string name = config["name"].asString();
    std::string pool_type = config["pool_type"].asString();
    int kernel_h = config["kernel"][0].asInt();
    int kernel_w = config["kernel"][1].asInt();
    int stride_h = config["stride"][0].asInt();
    int stride_w = config["stride"][1].asInt();
    int padding_h = config["padding"][0].asInt();
    int padding_w = config["padding"][1].asInt();
    if(pool_type.find("global") != std::string::npos) {
        kernel_h = tensor->getDimensions().d[1];
        kernel_w = tensor->getDimensions().d[2];
        stride_h = stride_w = 1;
        padding_h = padding_w = 0;
    }
    PoolingType poolingType;
    if(pool_type.find("max") != std::string::npos) {
        poolingType = PoolingType::kMAX;
    } else if(pool_type.find("avg") != std::string::npos) {
        poolingType = PoolingType ::kAVERAGE;
    } else {
        assert(false);
    }
    auto pool = network->addPooling(*tensor, poolingType, DimsHW(kernel_h, kernel_w));
    pool->setStride(DimsHW(stride_h, stride_w));
    pool->setPadding(DimsHW(padding_h, padding_w));
    pool->setAverageCountExcludesPadding(false);
    tensor = pool->getOutput(0);
    node2tensor[name] = tensor;
}

void relu(INetworkDefinition *network, Json::Value config, std::map<std::string, ITensor*> &node2tensor) {
    ITensor *tensor = get_input(network, config["inputs"], node2tensor, config);
    std::string name = config["name"].asString();
    tensor = network->addActivation(*tensor, ActivationType::kRELU)->getOutput(0);
    node2tensor[name] = tensor;
}

void element(INetworkDefinition *network, Json::Value config, std::map<std::string, ITensor*> &node2tensor) {
    ITensor *tensor = get_input(network, config["inputs"], node2tensor, config);
    std::string name = config["name"].asString();
    node2tensor[name] = tensor;
}

void activation(INetworkDefinition *network, Json::Value config, std::map<std::string, ITensor*> &node2tensor) {
    ITensor *tensor = get_input(network, config["inputs"], node2tensor, config);
    std::string name = config["name"].asString();
    ActivationType at;
    if(config["act_type"].asString() == "relu") {
        at = ActivationType::kRELU;
    } else if(config["act_type"].asString() == "sigmoid") {
        at = ActivationType::kSIGMOID;
    } else if(config["act_type"].asString() == "tanh") {
        at = ActivationType::kTANH;
    } else {
        FatalError("");
    }
    tensor = network->addActivation(*tensor, at)->getOutput(0);
    node2tensor[name] = tensor;
}

void add_node(INetworkDefinition *network, Json::Value config, std::map<std::string, ITensor*> &node2tensor);
void sequential(INetworkDefinition *network, Json::Value config, std::map<std::string, ITensor*> &node2tensor) {
    auto node_list = config["nodes"];
    int n = node_list.size();
    for(int i = 0; i < n; i++) {
        add_node(network, node_list[i], node2tensor);
        if(i == n - 1) {
            node2tensor[config["name"].asString()] = node2tensor[node_list[i]["name"].asString()];
        }
    }
}

void add_node(INetworkDefinition *network, Json::Value config, std::map<std::string, ITensor*> &node2tensor) {
    std::string type = config["type"].asString();
    if(type == "conv") {
        conv2d(network, config, node2tensor);
    } else if(type == "pool") {
        pool2d(network, config, node2tensor);
    } else if(type == "identity") {
        identity(network, config, node2tensor);
    } else if(type == "relu") {
        relu(network, config, node2tensor);
    } else if(type == "sequential") {
        sequential(network, config, node2tensor);
    } else if(type == "element") {
        element(network, config, node2tensor);
    } else if(type == "activation") {
        activation(network, config, node2tensor);
    } else {
        fprintf(stderr, "not support node type %s\n", type.c_str());
        assert(false);
    }
}

std::string get_output_name(Json::Value graph_config) {
    Json::Value blocks = graph_config["blocks"];
    return blocks[blocks.size()-1]["exit_node"]["name"].asString();
}

ICudaEngine *build_engine(Json::Value graph_config, int batch_size) {
    IBuilder *builder = createInferBuilder(gLogger);
    INetworkDefinition *network = builder->createNetwork();

    std::map<std::string, ITensor *> node2tensor;

    // add inputs
    placeholder(network, graph_config["input"], node2tensor);
    node2tensor[graph_config["input"]["name"].asString()]->setName("input");
    for (Json::Value block : graph_config["blocks"]) {
        for (const Json::Value &node : block["inner_nodes"]) {
            add_node(network, node, node2tensor);
        }
        add_node(network, block["exit_node"], node2tensor);
    }
    ITensor *output = node2tensor[get_output_name(graph_config)];
    output->setName("output");
    network->markOutput(*output);

    builder->setMaxBatchSize(batch_size);
    builder->setMaxWorkspaceSize(1 << 30);
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    network->destroy();
    builder->destroy();
    return engine;
}

extern "C" {
DLL void graph_latency(const char *graph_json, int batch_size, int warmup, int number, int repeat, float *results) {
    Json::Value graph_config;
    std::stringstream in(graph_json);
    in >> graph_config;

    weightManager.init(0, nullptr, nullptr, nullptr);
    ICudaEngine *engine = build_engine(graph_config, batch_size);
    IExecutionContext* context = engine->createExecutionContext();
    context->setProfiler(&gProfiler);

    int nbBindings = engine->getNbBindings();
    assert(nbBindings == 2);
    std::vector<void*> buffers(nbBindings);
    for (int i = 0; i < nbBindings; ++i) {
        Dims dims = engine->getBindingDimensions(i);
        size_t size = sizeof(float) * batch_size * dims.d[0] * dims.d[1] * dims.d[2];
        buffers[i] = safeCudaMalloc(size);
    }

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));
    for(int i = 0; i < warmup; i++)
        context->execute(batch_size, buffers.data());
    for (int i = 0; i < repeat; i++) {
        results[i] = 0.0;
        for(int j = 0; j < number; j++) {
            auto t_start = std::chrono::high_resolution_clock::now();
            context->enqueue(batch_size, buffers.data(), stream, nullptr);
            auto t_end = std::chrono::high_resolution_clock::now();
            results[i] += std::chrono::duration<float, std::milli>(t_end - t_start).count();
        }
        results[i] /= float(number);
    }

    for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx) {
        checkCUDA(cudaFree(buffers[bindingIdx]));
    }

    context->destroy();
    engine->destroy();
    checkCUDA(cudaStreamDestroy(stream));
}

DLL void graph_inference(const char *graph_json, int batch_size, float *input,
                         int num_convs, char **conv_names, float **filter_data, float **bias_data,
                         float *output) {
    Json::Value graph_config;
    std::stringstream in(graph_json);
    in >> graph_config;


    weightManager.init(num_convs, conv_names, filter_data, bias_data);
    ICudaEngine *engine = build_engine(graph_config, batch_size);
    IExecutionContext* context = engine->createExecutionContext();
    context->setProfiler(&gProfiler);

    int nbBindings = engine->getNbBindings();
    assert(nbBindings == 2);
    std::vector<void*> buffers(nbBindings);
    std::vector<size_t> sizes(nbBindings);
    for (int i = 0; i < nbBindings; ++i) {
        Dims dims = engine->getBindingDimensions(i);
        sizes[i] = sizeof(float) * batch_size * dims.d[0] * dims.d[1] * dims.d[2];
        buffers[i] = safeCudaMalloc(sizes[i]);
    }

    int input_index = engine->getBindingIndex("input");
    int output_index = engine->getBindingIndex("output");
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));
    checkCUDA(cudaMemcpyAsync(buffers[input_index], input, sizes[input_index], cudaMemcpyHostToDevice, stream));
    context->enqueue(batch_size, buffers.data(), stream, nullptr);
    checkCUDA(cudaMemcpyAsync(output, buffers[output_index], sizes[output_index], cudaMemcpyDeviceToHost, stream));
    checkCUDA(cudaDeviceSynchronize());
    for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
        checkCUDA(cudaFree(buffers[bindingIdx]));
    context->destroy();
    engine->destroy();
    checkCUDA(cudaStreamDestroy(stream));
}
}


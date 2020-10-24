//
// Created by yaoyao on 9/2/19.
//
#include <sstream>
#include <iostream>
#include <cudnn.h>
#include <cuda.h>
#include <assert.h>


#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      std::stringstream _error;                                        \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCUDA(status) do {                                         \
    if (status != 0) {                                                 \
      std::stringstream _error;                                        \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

struct CudnnContext {
    cudaStream_t stream;
    cudnnHandle_t dnn;
    size_t size;
    float *workspace;

    void init(size_t size) {
        this->size = size;
        checkCUDA(cudaStreamCreate(&stream));
        checkCUDNN(cudnnCreate(&dnn));
        checkCUDNN(cudnnSetStream(dnn, stream));
        checkCUDA(cudaMalloc(&workspace, size));
    }
    void unmap() {
        checkCUDA(cudaFree(workspace));
        checkCUDNN(cudnnDestroy(dnn));
        checkCUDA(cudaStreamDestroy(stream));
    }
};

struct Tensor {
    float *data;
    int batch_size;
    int channels;
    int height;
    int width;

    void init(int batch_size, int channels, int height, int width) {
        this->batch_size = batch_size;
        this->channels = channels;
        this->height = height;
        this->width = width;
    }
    void map() {
        size_t size = sizeof(float) * batch_size * channels * height * width;
        checkCUDA(cudaMalloc(&data, size));
    }
    void unmap() {
        checkCUDA(cudaFree(data));
    }
};

struct Conv2d {
    CudnnContext * context;
    Tensor *input;
    Tensor *output;
    int channels;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int padding_h;
    int padding_w;

    cudnnConvolutionFwdAlgo_t conv_alg;
    cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnActivationDescriptor_t actiDesc;

    float *filter_data;
    float *bias_data;

    void init(CudnnContext * context, Tensor *input, int channels, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w) {
        this->context = context;
        this->input = input;
        this->channels = channels;
        this->kernel_h = kernel_h;
        this->kernel_w = kernel_w;
        this->stride_h = stride_h;
        this->stride_w = stride_w;
        this->padding_h = padding_h;
        this->padding_w = padding_w;
    }
    void map() {
        output = new Tensor();
        output->init(input->batch_size, channels, 1 + (input->height - kernel_h + 2 * padding_h) / stride_h, 1 + (input->width - kernel_w + 2 * padding_w) / stride_w);
        output->map();
        checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&biasTensor));
        checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
        checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
        checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
        checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input->batch_size, input->channels, input->height, input->width));
        checkCUDNN(cudnnSetTensor4dDescriptor(biasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, channels, 1, 1));
        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, channels, input->channels, kernel_h, kernel_w));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding_h, padding_w, stride_h, stride_w, 1/*dilationH*/, 1/*dilationW*/, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        int n, c, h, w;
        checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, inputTensor, filterDesc, &n, &c, &h, &w));
        assert(n == output->batch_size);
        assert(c == output->channels);
        assert(h == output->height);
        assert(w == output->width);
        checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
        checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
        checkCUDNN(cudnnSetActivationDescriptor(actiDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));

        size_t filter_size = sizeof(float) * output->channels * input->channels * kernel_h * kernel_w;
        size_t bias_size = sizeof(float) * output->channels;
        checkCUDA(cudaMalloc(&filter_data, filter_size));
        checkCUDA(cudaMalloc(&bias_data, bias_size));

        const char *names[] = {
                "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
                "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
                "CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
                "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
                "CUDNN_CONVOLUTION_FWD_ALGO_FFT",
                "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
                "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
                "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED"
        };
        cudnnConvolutionFwdAlgoPerf_t perf[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
        int returned;
        checkCUDNN(cudnnFindConvolutionForwardAlgorithmEx(context->dnn, inputTensor, input->data, filterDesc,
                                               filter_data, convDesc, outputTensor, output->data,
                                               CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &returned, perf,
                                               context->workspace, context->size));
        this->conv_alg = perf[0].algo;
        fprintf(stderr, "%s\n", names[this->conv_alg]);
    }
    void forward() {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        checkCUDNN(cudnnConvolutionBiasActivationForward(
                context->dnn, &alpha, inputTensor, input->data, filterDesc, filter_data,
                convDesc, conv_alg, context->workspace, context->size,
                &beta, outputTensor, output->data, biasTensor, bias_data, actiDesc,
                outputTensor, output->data));
    }
    void unmap() {
        checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(biasTensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
        checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
        checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
        checkCUDNN(cudnnDestroyActivationDescriptor(actiDesc));
        // free tensors
        checkCUDA(cudaFree(filter_data));
        checkCUDA(cudaFree(bias_data));
        output->unmap();
    }
};


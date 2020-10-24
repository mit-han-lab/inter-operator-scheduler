#include <iostream>
#include <cstdio>
#include <cuda.h>
#include "utils.h"

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void print_kernel(int stage_id, int stream_id, int cnt) {
    CUDA_KERNEL_LOOP(i, cnt) {
        printf("%d %d %d\n", stage_id, stream_id, i);
    }
}

void test_event_synchronization() {
    const int NUM_STREAMS = 5;
    const int NUM_STAGES = 5;

    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t events[NUM_STAGES];

    for(int i = 0; i < NUM_STREAMS; i++) {
        checkCUDA(cudaStreamCreate(&streams[i]));
    }
    for(int i = 0; i < NUM_STAGES; i++) {
        checkCUDA(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming));
    }


    printf("HERE");
    for(int i = 0; i < NUM_STAGES; i++) {
        for(int j = 0; j < NUM_STREAMS; j++) {
            print_kernel<<<1, 1, 0, streams[j]>>>(i, j, NUM_STREAMS-j);
        }
//        cudaEventRecord(events[i], 0);
    }
    checkCUDA(cudaDeviceSynchronize());
}


int main() {
    test_event_synchronization();
}
/*
 *  code to measure latency, can also measure active warps pm
    void measure_latency(int warmup, int number, int repeat, float* results, bool profile_stage_latency, float *stage_results) {
        map();
        for(int i = 0; i < warmup; i++)
            forward();

        bool profile = false;
        volatile int test_complete;
		volatile int test_start;
        test_complete = 0;
		test_start = 0;
        testComplete = &test_complete;
		testStart = &test_start;
        if(profile) {
            int status;
            pthread_t pThread;
            eventName = "active_warps_pm";

            status = pthread_create(&pThread, NULL, sampling_func, NULL);
            if (status != 0) {
                perror("pthread_create");
                exit(-1);
            }
			while(*testStart == 0)
				usleep(1);
        }

        for(int i = 0; i < repeat; i++) {
            results[i] = 0.0;
			if(profile_stage_latency)
				for(int k = 0; k < num_stages; k++)
					stage_results[i * num_stages + k] = 0.0;
            for (int j = 0; j < number; j++) {
                auto sll = forward(profile_stage_latency);
				if(profile_stage_latency)
					for(int k = 0; k < num_stages; k++)
						stage_results[i * num_stages + k] += sll[k];
                results[i] += sll.back();
            }
			if(profile_stage_latency)
				for(int k = 0; k < num_stages; k++)
					stage_results[i * num_stages + k] /= float(number);
            results[i] /= float(number);
        }
        test_complete = 1;
        unmap();
    }
 */

//
// Created by yaoyao on 10/27/19.
//
#include <chrono>
#include <cupti_events.h>
#include <unistd.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include "ios/profile.h"

#define CHECK_CU_ERROR(err, cufunc)                                     \
  if (err != CUDA_SUCCESS)                                              \
    {                                                                   \
      printf ("Error %d for CUDA Driver API function '%s'.\n",          \
              err, cufunc);                                             \
      exit(-1);                                                         \
    }
#define CHECK_CUPTI_ERROR(err, cuptifunc)                       \
  if (err != CUPTI_SUCCESS)                                     \
    {                                                           \
      const char *errstr;                                       \
      cuptiGetResultString(err, &errstr);                       \
      printf ("%s:%d:Error %s for CUPTI API function '%s'.\n",  \
              __FILE__, __LINE__, errstr, cuptifunc);           \
      exit(-1);                                                 \
    }

volatile int *testComplete;
volatile int *testStart;
const char *eventName;
static CUcontext context;
static CUdevice device;
#define SAMPLE_PERIOD_MS 10

void init_profile()
{
    if(context != nullptr) return;
    CUresult err;
    int deviceNum = 0;
    err = cuInit(0);
    CHECK_CU_ERROR(err, "cuInit");
    err = cuDeviceGet(&device, deviceNum);
    CHECK_CU_ERROR(err, "cuDeviceGet");
    err = cuCtxCreate(&context, 0, device);
    CHECK_CU_ERROR(err, "cuCtxCreate");
}

void * sampling_func(void *arg)
{
    CUptiResult cuptiErr;
    CUpti_EventGroup eventGroup;
    CUpti_EventID eventId;
    size_t bytesRead, valueSize;
    uint32_t numInstances = 0, j = 0;
    uint64_t *eventValues = NULL, eventVal = 0;
    uint32_t profile_all = 1;

    cuptiErr = cuptiSetEventCollectionMode(context, CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS);
//    cuptiErr = cuptiSetEventCollectionMode(context, CUPTI_EVENT_COLLECTION_MODE_KERNEL);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiSetEventCollectionMode");

    cuptiErr = cuptiEventGroupCreate(context, &eventGroup, 0);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupCreate");

    cuptiErr = cuptiEventGetIdFromName(device, eventName, &eventId);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGetIdFromName");

    cuptiErr = cuptiEventGroupAddEvent(eventGroup, eventId);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupAddEvent");

    cuptiErr = cuptiEventGroupSetAttribute(eventGroup,
                                           CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
                                           sizeof(profile_all), &profile_all);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupSetAttribute");

    cuptiErr = cuptiEventGroupEnable(eventGroup);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupEnable");

    valueSize = sizeof(numInstances);
    cuptiErr = cuptiEventGroupGetAttribute(eventGroup,
                                           CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                                           &valueSize, &numInstances);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupGetAttribute");

    bytesRead = sizeof(uint64_t) * numInstances;
    eventValues = (uint64_t *) malloc(bytesRead);
    if (eventValues == NULL) {
        printf("%s:%d: Failed to allocate memory.\n", __FILE__, __LINE__);
        exit(-1);
    }

	std::vector<unsigned long long> vc;
    std::vector<unsigned long long> tvc;
	vc.reserve(5000);
	tvc.reserve(5000);
	*testStart = 1;

    auto begin = std::chrono::high_resolution_clock::now();

    while (!*testComplete) {
        cuptiErr = cuptiEventGroupReadEvent(eventGroup,
                                            CUPTI_EVENT_READ_FLAG_NONE,
                                            eventId, &bytesRead, eventValues);
        CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupReadEvent");
        if (bytesRead != (sizeof(uint64_t) * numInstances)) {
            printf("Failed to read value for \"%s\"\n", eventName);
            exit(-1);
        }

        for (j = 0; j < numInstances; j++) {
            eventVal += eventValues[j];
        }
		vc.push_back((unsigned long long)eventVal);
        tvc.push_back((unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()-begin).count());
//        printf("%s: %llu\n", eventName, (unsigned long long)eventVal);
#ifdef _WIN32
        Sleep(SAMPLE_PERIOD_MS);
#else
//        usleep(SAMPLE_PERIOD_MS * 1000);
        usleep(1);
#endif
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();

    std::cout << "Event Name: " << eventName << std::endl;
    std::cout << "#Stamps: " << vc.size() << std::endl;
    std::cout << "Duration: " << duration << " ns" << std::endl;
    std::cout << "Average duration between two stamps: " << duration / (vc.size()) << std::endl;

	for(int i = 0; i < vc.size(); i++)
		printf("%llu %llu\n", vc[i], tvc[i]);

    cuptiErr = cuptiEventGroupDisable(eventGroup);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupDisable");

    cuptiErr = cuptiEventGroupDestroy(eventGroup);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupDestroy");

    free(eventValues);
    return NULL;
}




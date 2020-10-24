//
// Created by yaoyao on 10/27/19.
//

#ifndef MUSIC_PROFILE_H
#define MUSIC_PROFILE_H

extern volatile int *testComplete;
extern volatile int *testStart;
extern const char *eventName;
void init_profile();
void * sampling_func(void *arg);

#endif //MUSIC_PROFILE_H

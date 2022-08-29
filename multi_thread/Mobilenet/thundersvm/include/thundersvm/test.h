  
// #ifndef TEST_H
// #define TEST_H

// #include "cuda_runtime.h"
// #include "curand.h"
// #include "cublas_v2.h"
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>
// #include <ATen/cuda/CUDAMultiStreamGuard.h>
// #include <ATen/cuda/CUDAEvent.h>
// #include <c10/cuda/CUDAStream.h>
// #include <c10/cuda/CUDAGuard.h>
// #include <pthread.h>
// #include "thpool.h"
// #include <utility>

// #include <thundersvm/util/log.h>
// #include <thundersvm/model/svc.h>
// #include <thundersvm/model/svr.h>
// #include <thundersvm/model/oneclass_svc.h>
// #include <thundersvm/model/nusvc.h>
// #include <thundersvm/model/nusvr.h>
// #include <thundersvm/util/metric.h>
// #include "thundersvm/cmdparser.h"
// #include "thundersvm/alex.h"
// #include "thundersvm/vgg.h"
// #include "thundersvm/resnet.h"
// #include "thundersvm/densenet.h"
// #include "thundersvm/squeeze.h"
// #include "thundersvm/mobile.h"
// #include "thundersvm/mnasnet.h"
// #include "thundersvm/inception.h"
// #include "thundersvm/shuffle.h"

// #define n_streamPerPool 32 // pytorch streampool 32개
// #define threshold 

// extern threadpool thpool; 
// extern pthread_cond_t cond_t;
// extern pthread_mutex_t mutex_t;
// extern int cond_i;
// // extern pthread_cond_t *cond_t;
// // extern pthread_mutex_t *mutex_t;
// // extern int *cond_i;
// extern pthread_cond_t *cond_t_vi;       /* when you want to build thundersvm-train erase it */
// extern pthread_mutex_t *mutex_t_vi;     /* when you want to build thundersvm-train erase it */
// extern int *cond_i_vi;

// extern pthread_cond_t *cond_t_vi_derived;   /* when you want to build thundersvm-train erase it */
// extern pthread_mutex_t *mutex_t_vi_derived; /* when you want to build thundersvm-train erase it */
// extern int *cond_i_vi_derived;              /* when you want to build thundersvm-train erase it */

// extern pthread_mutex_t *mutex_t_cond;

// extern int stream_priority;
// extern std::vector<std::vector <at::cuda::CUDAStream>> streams;
// extern cudaEvent_t event_A;

// #endif


//////////////////////////////////////////////순서를 바꿔서 시도해봅시다/////////////////////////////////////////////////////////////
  
#ifndef TEST_H
#define TEST_H

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAMultiStreamGuard.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <pthread.h>
#include "thpool.h"
#include "svm.h"
#include <utility>

#include <thundersvm/util/log.h>
#include <thundersvm/model/svc.h>
#include <thundersvm/model/svr.h>
#include <thundersvm/model/oneclass_svc.h>
#include <thundersvm/model/nusvc.h>
#include <thundersvm/model/nusvr.h>
#include <thundersvm/util/metric.h>
#include "thundersvm/cmdparser.h"


#define n_streamPerPool 32 // pytorch streampool 32개
#define threshold 
#define NUMVI 27

extern threadpool thpool; 
extern pthread_cond_t vi_cond_t, pi_cond_t;
extern pthread_mutex_t vi_mutex_t, pi_mutex_t;
extern int vi_cond_i, pi_cond_i;
extern pthread_cond_t* cond_t;
extern pthread_mutex_t* mutex_t;
extern int* cond_i;


extern c10::DeviceIndex GPU_NUM;
extern int stream_priority;
extern std::vector <at::cuda::CUDAStream> streams;
// extern std::vector<std::vector <at::cuda::CUDAStream>> streams;
extern cudaEvent_t event_A;
extern std::shared_ptr<SvmModel> vi_load(std::string vi_svm_path);
extern vector<Visvm> vi_svm;
extern vector<DM_Pisvm> pi_svm;
extern std::vector<at::Tensor> dm_out_list;
extern std::vector<bool> dm_out_check;

#endif
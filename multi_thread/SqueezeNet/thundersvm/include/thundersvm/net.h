// #ifndef NET_H
// #define NET_H

// #include <vector>
// #include <torch/torch.h>
// #include <string>
// #include <functional>
// #include "cuda_runtime.h"

// #include "curand.h"
// #include "cublas_v2.h"


// struct Dummy : torch::jit::Module{};

// /* hidden queue */
// typedef struct output{
// 	struct output* next;                   /* pointer to previous output   */                         /* function's argument       */
// 	torch::jit::IValue data;
// 	// at::Tensor data;
// } output;

// typedef struct hiddenqueue{
//     output *front;
//     output *rear;
//     int count;
// } hiddenqueue;

// typedef struct _layer
// {
// 	at::Tensor output;
// 	std::string name;	//layer name
// 	torch::jit::Module layer;
// 	bool exe_success;	//layer operation complete or not
// 	std::vector<int> from_idx;	//concat
// 	std::vector<int> branch_idx;	// last layer idx of branch for eventrecord
// 	int input_idx; 	//network with branch
// 	int event_idx;	//network with branch
// 	int skip;	//inception skip num in a branch
// 	int stream_idx;	//stream index of current layers
// }Layer;

// typedef struct _net
// {
// 	std::vector<Layer> layers;
// 	std::vector<torch::jit::IValue> input;
// 	at::Tensor identity;	//resnet
// 	std::vector<cudaEvent_t> record;
// 	std::vector<at::Tensor> chunk; //shuffle
// 	std::string name;	//network name
// 	int index; //layer index
// 	int index_n; // network index
// 	int index_s; // stream index
// 	int index_b; // index of stream for branch layer 
// 	int n_all; // all network num
// 	int flatten; //flatten layer index
// 	int priority; // priQ priority
// 	int H_L; // 0 = HIGH , 1 = LOW
// 	hiddenqueue *hiddenqueue_p;
// }Net;

// typedef struct _netlayer
// {
// 	Net *net;
// }netlayer;

// typedef struct vi_svm{ 				/* struct for vi_svm */
// 	int index_n; 					/* for signal */
// 	hiddenqueue *hiddenqueue_p; 	/* for pull from the Q */
// 	char *test_file_name[];
// }vi_svm;

// typedef struct derived_model{		/* struct for derived_model */
// 	char hidden_output_name[128];		/* input file of derived model */
// 	int index_n;						/* struct index */
// 	bool exe_success;					/* flag of derived model */
// 	// at::Tensor derived_output;
// 	torch::jit::IValue derived_output;
// 	torch::jit::Module derived_module;	/* derived model */
// }derived_model;

// /* hidden layer output queue prototype */
// hiddenqueue *hiddenqueue_init(void);
// int hiddenqueue_empty(hiddenqueue *hiddenqueue_p);
// void hiddenqueue_push(hiddenqueue *hiddenqueue_p, torch::jit::IValue data);
// // void hiddenqueue_push(hiddenqueue *hiddenqueue_p, at::Tensor data);
// torch::jit::IValue hiddenqueue_pull(hiddenqueue *hiddenqueue_p);
// // at::Tensor hiddenqueue_pull(hiddenqueue *hiddenqueue_p);
// int hiddenqueue_size(hiddenqueue *hiddenqueue_p);
// /* hidden layer output queue prototype */

// #endif

//////////////////////////////////////////순서를 바꿔서 시도해봅시다////////////////////////////////////////////////////////

#ifndef NET_H
#define NET_H

#include <vector>
#include <torch/torch.h>
#include <string>
#include <functional>
#include <iostream>
#include "cuda_runtime.h"

#include "curand.h"
#include "cublas_v2.h"

/* thunder */
#include <thundersvm/util/log.h>
#include <thundersvm/cmdparser.h>
#include <thundersvm/model/svmmodel.h>
#include <thundersvm/model/nusvr.h>
#include <thundersvm/model/svc.h>
#include <thundersvm/model/nusvc.h>
#include <thundersvm/model/oneclass_svc.h>
#include <thundersvm/util/metric.h>
#include "thundersvm/svm.h"
/* thunder */
//using namespace std;
struct Dummy : torch::jit::Module{};

/* hidden queue */
typedef struct output{
	struct output* next;                   /* pointer to previous output   */                         /* function's argument       */
	torch::jit::IValue data;
	// at::Tensor data;
} output;

typedef struct hiddenqueue{
    output *front;
    output *rear;
    int count;
} hiddenqueue;

typedef struct _layer
{
	at::Tensor output;
	// torch::jit::IValue output;
	std::string name;	//layer name
	torch::jit::Module layer;
	bool exe_success;	//layer operation complete or not
	std::vector<int> from_idx;	//concat
	std::vector<int> branch_idx;	// last layer idx of branch for eventrecord
	int input_idx; 	//network with branch
	int event_idx;	//network with branch
	int skip;	//inception skip num in a branch
	int stream_idx;	//stream index of current layers
}Layer;

typedef struct Visvm	/* vi svm struct */
{
	at::Tensor hidden_out;
	// torch::jit::IValue hidden_out;
	int index_n;
	std::shared_ptr<SvmModel> model;
}Visvm;

typedef struct DM_Pisvm{
	torch::jit::Module derived_module;
	// torch::jit::IValue hidden_out;
	at::Tensor hidden_out;
	std::shared_ptr<SvmModel> model;
	int index_n;

}DM_Pisvm;

typedef struct _net
{
	std::vector<Layer> layers;
	std::vector<torch::jit::IValue> input;
	std::vector<torch::jit::IValue> input1;
	std::vector<torch::jit::IValue> input2;
	at::Tensor identity;	//resnet
	std::vector<cudaEvent_t> record;
	std::vector<at::Tensor> chunk; //shuffle
	std::string name;	//network name
	int index; //layer index
	int index_n; // network index
	int index_s; // stream index
	int index_b; // index of stream for branch layer 
	int n_all; // all network num
	int flatten; //flatten layer index
	int priority; // priQ priority
	int H_L; // 0 = HIGH , 1 = LOW
	std::vector<int> stream_id;
	hiddenqueue *hiddenqueue_p;
	Visvm *visvm_p;
}Net;

typedef struct _netlayer
{
	Net *net;
}netlayer;


/* hidden layer output queue prototype */
hiddenqueue *hiddenqueue_init(void);
int hiddenqueue_empty(hiddenqueue *hiddenqueue_p);
void hiddenqueue_push(hiddenqueue *hiddenqueue_p, torch::jit::IValue data);
// void hiddenqueue_push(hiddenqueue *hiddenqueue_p, at::Tensor data);
torch::jit::IValue hiddenqueue_pull(hiddenqueue *hiddenqueue_p);
// at::Tensor hiddenqueue_pull(hiddenqueue *hiddenqueue_p);
int hiddenqueue_size(hiddenqueue *hiddenqueue_p);
/* hidden layer output queue prototype */

#endif

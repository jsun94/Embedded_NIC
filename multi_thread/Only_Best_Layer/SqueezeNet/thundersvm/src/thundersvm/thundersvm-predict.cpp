//
// Created by jiashuai on 17-10-31.
//

#include <thundersvm/util/log.h>
#include <thundersvm/cmdparser.h>
#include <thundersvm/model/svmmodel.h>
#include <thundersvm/model/nusvr.h>
#include <thundersvm/model/svc.h>
#include <thundersvm/model/nusvc.h>
#include <thundersvm/model/oneclass_svc.h>
#include <thundersvm/util/metric.h>
#ifdef _WIN32
INITIALIZE_EASYLOGGINGPP
#endif
using std::fstream;


//////////test.cpp header files//////////
#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <stdio.h>
#include <memory>
#include <stdlib.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAFunctions.h>
#include <time.h>
#include <string>
#include <unistd.h>
#include "thundersvm/squeeze.h"


#include "thundersvm/test.h"

using std::fstream;



#define n_threads 8 // inception의 병렬화 실행을 위한 최소한의 thread 갯수
#define WARMING 4
#define NUMVI 11   /* number of vi_svm */
#define NUMPI 2
#define ALEX_FLATTEN 5

extern void *predict_squeeze(Net *input);
extern void *predict_squeeze_warm(Net *input);

namespace F = torch::nn::functional;
//using namespace std;
using std::cout;
using std::endl;
using std::cerr;




/* vi svm model load function */
std::shared_ptr<SvmModel> vi_load(std::string vi_svm_path){
	std::fstream file;
	file.open(vi_svm_path, std::fstream::in);

	string feature, svm_type;
	// file >> feature >> svm_type;
	// CHECK_EQ(feature, "svm_type");
	svm_type = "one_class";
	std::shared_ptr<SvmModel> model;
	if (svm_type == "one_class") {
		model.reset(new OneClassSVC());
	}            

	CUDA_CHECK(cudaSetDevice(0));

	model->set_max_memory_size_Byte(8192*10);
	model->load_from_file(vi_svm_path);
	file.close(); 
	return model;
}
/* vi svm model load function */


threadpool thpool;
pthread_cond_t vi_cond_t, pi_cond_t;
pthread_mutex_t vi_mutex_t, pi_mutex_t;
int vi_cond_i, pi_cond_i;
pthread_cond_t* cond_t;
pthread_mutex_t* mutex_t;
int* cond_i;


// std::vector<std::vector <at::cuda::CUDAStream>> streams;
std::vector <at::cuda::CUDAStream> streams;
c10::DeviceIndex GPU_NUM = 0;
/////from test.cpp end//////////
std::vector<Visvm> vi_svm;  /* extern */
std::vector<DM_Pisvm> pi_svm; /* extern */
std::vector<at::Tensor> dm_out_list(NUMPI + 1);
std::vector<bool> dm_out_check(NUMPI + 1, false);


int main(int argc, char **argv) {
	c10::cuda::set_device(GPU_NUM);
	torch::Device device = {at::kCUDA,GPU_NUM};

	// std::string filename = argv[2];
	int n_squeeze=1;
	float time;
	cudaEvent_t start, end;

	int n_all = n_squeeze;

	static int stream_index_H = 0;
	static int branch_index_H = 31;


	for(int i=0; i<n_streamPerPool; i++){
	streams.push_back(at::cuda::getStreamFromPool(true,GPU_NUM));
	}

	thpool = thpool_init(n_threads, n_all);

	torch::jit::script::Module squeezeModule;

	try {
		squeezeModule = torch::jit::load("/home/nvidia/joo/models/SqueezeNet/squeezenet_jit_cifar10_test.pt", at::kCUDA);
		squeezeModule.to(device);
	}
	catch (const c10::Error& e) {
	cerr << "error loading the model\n";
	return -1;
	}
	cout<<"***** Model Load compelete *****"<<"\n";

	cond_t = (pthread_cond_t *)malloc(sizeof(pthread_cond_t) * n_all);
	mutex_t = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) * n_all);
	cond_i = (int *)malloc(sizeof(int) * n_all);

	for (int i = 0; i < n_all; i++)
	{
		pthread_cond_init(&cond_t[i], NULL);
		pthread_mutex_init(&mutex_t[i], NULL);
		cond_i[i] = 0;
	}


//   /* vi svm path */
// 	std::string vi_svm_path_list[NUMVI] = {
// 		"/home/nvidia/joo/models/SqueezeNet/VISVM/layer65_visvm.bin",
// 		"/home/nvidia/joo/models/SqueezeNet/VISVM/layer70_visvm.bin",
// 		"/home/nvidia/joo/models/SqueezeNet/VISVM/layer72_visvm.bin",
// 		"/home/nvidia/joo/models/SqueezeNet/VISVM/layer73_visvm.bin"
// 	};
// 	std::string pi_svm_path_list[NUMPI] = {
// 		"/home/nvidia/joo/models/SqueezeNet/PISVM/max_layer65_layer70_pisvm.bin",
// 		"/home/nvidia/joo/models/SqueezeNet/PISVM/max_layer70_layer72_pisvm.bin",
// 		"/home/nvidia/joo/models/SqueezeNet/PISVM/max_layer72_layer73_pisvm.bin"
// 	};
// 	char* derived_model_path_list[NUMVI] = {
// 		"/home/nvidia/joo/models/SqueezeNet/D_M/MT/layer65_derived_model.pt",
// 		"/home/nvidia/joo/models/SqueezeNet/D_M/MT/layer70_derived_model.pt",
// 		"/home/nvidia/joo/models/SqueezeNet/D_M/MT/layer72_derived_model.pt",
// 		"/home/nvidia/joo/models/SqueezeNet/D_M/MT/layer73_derived_model.pt"
// 	};

  /* vi svm path */
	std::string vi_svm_path_list[NUMVI] = {
		"/home/nvidia/joo/models/SqueezeNet/VISVM/layer7_visvm.bin",
		"/home/nvidia/joo/models/SqueezeNet/VISVM/layer15_visvm.bin",
		"/home/nvidia/joo/models/SqueezeNet/VISVM/layer23_visvm.bin",
		"/home/nvidia/joo/models/SqueezeNet/VISVM/layer32_visvm.bin",
		"/home/nvidia/joo/models/SqueezeNet/VISVM/layer40_visvm.bin",
		"/home/nvidia/joo/models/SqueezeNet/VISVM/layer48_visvm.bin",
		"/home/nvidia/joo/models/SqueezeNet/VISVM/layer56_visvm.bin",
		"/home/nvidia/joo/models/SqueezeNet/VISVM/layer62_visvm.bin",
		"/home/nvidia/joo/models/SqueezeNet/VISVM/layer65_visvm.bin",
		"/home/nvidia/joo/models/SqueezeNet/VISVM/layer70_visvm.bin",
		"/home/nvidia/joo/models/SqueezeNet/VISVM/layer72_visvm.bin"
	};
	std::string pi_svm_path_list[NUMPI] = {
		"/home/nvidia/joo/models/SqueezeNet/PISVM/layer3_layer4_pisvm.bin",
		"/home/nvidia/joo/models/SqueezeNet/PISVM/layer4_layer7_pisvm.bin"
	};
	char* derived_model_path_list[NUMPI+1] = {
		"/home/nvidia/joo/models/SqueezeNet/D_M/MT/layer3_derived_model_jit.pt",
		"/home/nvidia/joo/models/SqueezeNet/D_M/MT/layer4_derived_model_jit.pt",
		"/home/nvidia/joo/models/SqueezeNet/D_M/MT/layer7_derived_model_jit.pt"
	};

	/* svm WARM UP input data */
	at::Tensor warm_data = at::zeros({1,1});
	DataSet warm_data_input;
	float warm_label = 0.0;
	warm_data_input.load_from_dense(1, warm_data.sizes()[0], warm_data.data<float>(), &warm_label);
	/* svm WARM UP input data */

	/* derived model WARM UP */
	int dm_warmup_shape[NUMPI + 1] = {98304, 24576, 4096};

	// for (int i = 0; i < NUMVI; i++){
	// 	Visvm v;
	// 	v.index_n = i;
	// 	v.model = vi_load(vi_svm_path_list[i]);
	// 	/* visvm WARMUP */
	// 	v.model->predict_with_core(warm_data_input.instances(),-1, 0, 0);
	// 	vi_svm.push_back(v);
	// 	std::cout << i << " done" << std::endl;
	// }
	// std::cout << "print : " << vi_svm[0].model->get_n_sv() << std::endl;

  /* derived model path */
//   /* Derived model name list */

	for (int i = 0; i < (NUMPI + 1); i++){
		DM_Pisvm d_p;
		d_p.index_n = i;
		d_p.derived_module = torch::jit::load(derived_model_path_list[i], device);
		// d_p.derived_module.to(device);
		/* derived model WARM UP */
		std::vector<torch::jit::IValue> dm_warmup_vector;
		at::Tensor dm_warmup_input = torch::ones({1, dm_warmup_shape[i]}).to(at::kCUDA);
		dm_warmup_vector.push_back(dm_warmup_input);
		d_p.derived_module.forward(dm_warmup_vector).toTensor();

		if(i != 0){
			d_p.model = vi_load(pi_svm_path_list[i-1]);
			/*pisvm WARMUP */
			d_p.model->predict_with_core(warm_data_input.instances(),-1, 1, 1);
	}
    //skycs
    //d_p.model = svm_load_model(pi_svm_path_list[i]);

    pi_svm.push_back(d_p);
    std::cout << i << " done2" << std::endl;
	}	

	pthread_cond_init(&vi_cond_t, NULL);
	pthread_cond_init(&pi_cond_t, NULL);
	pthread_mutex_init(&vi_mutex_t, NULL);
	pthread_mutex_init(&pi_mutex_t, NULL);
	pi_cond_i = vi_cond_i = 1;



	vector<torch::jit::IValue> inputs;
	//test data random
	torch::Tensor x = torch::ones({1, 3, 32, 32}).to(at::kCUDA); //CIFAR10
	inputs.push_back(x);
	//test data random

	// //test data for 100 adv sample
	// torch::jit::script::Module container = torch::jit::load("/home/nvidia/joo/models/fgsm_jit.pt");
	// torch::Tensor x = container.attr("a").toTensor().to(at::kCUDA);
	// std::cout << typeid(x).name() << std::endl;
	// std::cout << x.sizes() << std::endl;
	// inputs.push_back(x);
	// //test data for 100 adv sample

	Net net_input_squeeze[n_squeeze];
	pthread_t networkArray_squeeze[n_squeeze];



	for(int i=0;i<n_squeeze;i++){
		net_input_squeeze[i].index_n = i;
		get_submodule_squeeze(squeezeModule, net_input_squeeze[i]);
		std::cout << "End get submodule_squeezenet " << i <<"\n";
		net_input_squeeze[i].input = inputs;
		net_input_squeeze[i].name = "SqueezeNet";
		// net_input_alex[i].flatten = net_input_alex[i].layers.size() - ALEX_FLATTEN;
		net_input_squeeze[i].stream_id = {stream_index_H%n_streamPerPool};
		stream_index_H+=1;
		/*=============WARM UP FOR OPTIMIZATION===============*/
		/*=============FILE===============*/
		//net_input_alex[i].fp = fopen((filename+"-"+"A"+".txt").c_str(),"a");
		for (int j = 0; j < 4; j++){
			predict_squeeze_warm(&net_input_squeeze[i]);
			net_input_squeeze[i].input = inputs;
		}
	}


	std::cout<<"\n==================WARM UP END==================\n";

/* time check */
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);


	for(int i=0;i<n_squeeze;i++){
	if (pthread_create(&networkArray_squeeze[i], NULL, (void *(*)(void*))predict_squeeze, &net_input_squeeze[i]) < 0){
		perror("thread error");
		exit(0);
	}
	}


	for (int i = 0; i < n_squeeze; i++){
	pthread_join(networkArray_squeeze[i], NULL);
	}

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);
	std::cout << "Total EXE TIME = "<< time/1000<<"'s"<< std::endl;
	/* resX time check end */
	sleep(10);
	cudaDeviceSynchronize();
	free(cond_t);
	free(mutex_t);
	free(cond_i);
}

                            
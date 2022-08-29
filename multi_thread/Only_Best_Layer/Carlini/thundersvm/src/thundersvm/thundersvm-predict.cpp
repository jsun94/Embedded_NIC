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
#include "thundersvm/carlini.h"

#include "thundersvm/test.h"
#include <ctime>

using std::fstream;


#define n_carlini 1

#define n_threads 8 // inception의 병렬화 실행을 위한 최소한의 thread 갯수
#define WARMING 4
#define NUMVI 1   /* number of vi_svm */
#define NUMPI 8

extern void *predict_carlini(Net *input);
extern void *predict_carlini_warm(Net *input);

namespace F = torch::nn::functional;
//using namespace std;
using std::cout;
using std::endl;
using std::cerr;

void print_script_module(const torch::jit::script::Module& module, size_t spaces) {
    for (const auto& sub_module : module.named_children()) {
        if (!sub_module.name.empty()) {
            std::cout << std::string(spaces, ' ') << sub_module.value.type()->name().value().name() << " " << sub_module.name << "\n";    
        }
        print_script_module(sub_module.value, spaces + 2);
    }
}

void print_vector(vector<int> v){
	for(int i=0;i<v.size();i++){
		cout<<v[i]<<" ";
	}
	cout<<"\n";
}


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

std::vector<at::cuda::CUDAStream> streams;
int16_t GPU_NUM = 0;
/////from test.cpp end//////////
std::vector<Visvm> vi_svm;  /* extern */
std::vector<DM_Pisvm> pi_svm; /* extern */
std::vector<at::Tensor> dm_out_list(NUMPI+1);
std::vector<bool> dm_out_check(NUMPI+1, false);


int main(int argc, char **argv) {
	c10::cuda::set_device(GPU_NUM);
	torch::Device device = {at::kCUDA,GPU_NUM};
	int n_all = n_carlini;

	float timer;
	cudaEvent_t start, end;

	thpool = thpool_init(n_threads, n_all);

  /* stream 생성 */

	for(int i=0; i<n_streamPerPool; i++){
		streams.push_back(at::cuda::getStreamFromPool(true,0)); //low priority stream  (priority = 0)
	}


	torch::jit::script::Module carliniModule;

/* Model Load */
	try {
		carliniModule = torch::jit::load("/home/nvidia/joo/models/Carlini/carlini_cifar10_jit.pt",device);
		carliniModule.to(device);
	}
	catch (const c10::Error& e) {
		cerr << "error loading the model\n";
		return -1;
	}
	cout<<"***** Model Load compelete *****"<<"\n";

  /* vi svm path */
	std::string vi_svm_path_list[NUMVI] = {
		"/home/nvidia/joo/models/Carlini/VISVM/layer5_visvm.bin"
	};
	// std::string pi_svm_path_list[NUMPI] = {
	// 	"/home/nvidia/joo/models/Carlini/PISVM/layer2_layer4_pisvm.bin",
	// 	"/home/nvidia/joo/models/Carlini/PISVM/layer4_layer5_pisvm.bin",
	// 	"/home/nvidia/joo/models/Carlini/PISVM/layer5_layer7_pisvm.bin",
	// 	"/home/nvidia/joo/models/Carlini/PISVM/layer7_layer9_pisvm.bin",
	// 	"/home/nvidia/joo/models/Carlini/PISVM/layer9_layer10_pisvm.bin",
	// 	"/home/nvidia/joo/models/Carlini/PISVM/layer10_layer11_pisvm.bin",
	// 	"/home/nvidia/joo/models/Carlini/PISVM/layer11_layer13_pisvm.bin",
	// 	"/home/nvidia/joo/models/Carlini/PISVM/layer13_layer15_pisvm.bin"
	// };
	// char* derived_model_path_list[NUMPI+1] = {
	// 	"/home/nvidia/joo/models/Carlini/D_M/MT/layer2_derived_model_jit.pt",
	// 	"/home/nvidia/joo/models/Carlini/D_M/MT/layer4_derived_model_jit.pt",
	// 	"/home/nvidia/joo/models/Carlini/D_M/MT/layer5_derived_model_jit.pt",
	// 	"/home/nvidia/joo/models/Carlini/D_M/MT/layer7_derived_model_jit.pt",
	// 	"/home/nvidia/joo/models/Carlini/D_M/MT/layer9_derived_model_jit.pt",
	// 	"/home/nvidia/joo/models/Carlini/D_M/MT/layer10_derived_model_jit.pt",
	// 	"/home/nvidia/joo/models/Carlini/D_M/MT/layer11_derived_model_jit.pt",
	// 	"/home/nvidia/joo/models/Carlini/D_M/MT/layer13_derived_model_jit.pt",
	// 	"/home/nvidia/joo/models/Carlini/D_M/MT/layer15_derived_model_jit.pt"
	// };

	/* svm WARM UP input data */
	at::Tensor warm_data = at::zeros({1,1});
	DataSet warm_data_input;
	float warm_label = 0.0;
	warm_data_input.load_from_dense(1, warm_data.sizes()[0], warm_data.data<float>(), &warm_label);
	/* svm WARM UP input data */

	/* derived model WARM UP */
	int dm_warmup_shape[NUMPI+1] = {57600, 50176, 12544, 18432, 12800, 3200, 3200, 256, 256};
	// int dm_warmup_shape[NUMPI+1] = {12800, 3200, 3200, 256, 256};

	for (int i = 0; i < NUMVI; i++){
		Visvm v;
		v.index_n = i;
		v.model = vi_load(vi_svm_path_list[i]);
		/* visvm WARMUP */
		v.model->predict_with_core(warm_data_input.instances(),-1, 0, 0);
		vi_svm.push_back(v);
		std::cout << i << " done" << std::endl;
	}


  /* derived model path */
//   /* Derived model name list */

	// for (int i = 0; i < NUMPI+1; i++){
	// 	DM_Pisvm d_p;
	// 	d_p.index_n = i;
	// 	d_p.derived_module = torch::jit::load(derived_model_path_list[i],device);
	// 	std::vector<torch::jit::IValue> dm_warmup_vector;
	// 	at::Tensor dm_warmup_input = torch::ones({1, dm_warmup_shape[i]}).to(at::kCUDA);
	// 	dm_warmup_vector.push_back(dm_warmup_input);
	// 	d_p.derived_module.forward(dm_warmup_vector).toTensor();
	// 	if(i != 0){
	// 		d_p.model = vi_load(pi_svm_path_list[i-1]);
	// 		/*pisvm WARMUP */
	// 		d_p.model->predict_with_core(warm_data_input.instances(),-1, 1, 1);
	// }
    // //skycs
    // //d_p.model = svm_load_model(pi_svm_path_list[i]);

    // pi_svm.push_back(d_p);
    // std::cout << i << " done2" << std::endl;
	// }	

	pthread_cond_init(&vi_cond_t, NULL);
	pthread_cond_init(&pi_cond_t, NULL);
	pthread_mutex_init(&vi_mutex_t, NULL);
	pthread_mutex_init(&pi_mutex_t, NULL);
	pi_cond_i = vi_cond_i = 1;



	vector<torch::jit::IValue> inputs;

	torch::Tensor x = torch::ones({1, 3, 32, 32}).to(device);
	inputs.push_back(x);

	Net net_input_carlini[n_carlini];
	pthread_t networkArray_carlini[n_carlini];



	for(int i=0;i<n_carlini;i++){
		get_submodule_carlini(carliniModule, net_input_carlini[i]);
		std::cout << "End get submodule_carlini "<< i << "\n";
		net_input_carlini[i].input = inputs;
		net_input_carlini[i].name = "Carlini";
		net_input_carlini[i].flatten = net_input_carlini[i].layers.size()-1;
		net_input_carlini[i].index_n = i;

		for(int j = 0; j < 3; j++){
			predict_carlini_warm(&net_input_carlini[i]);
			net_input_carlini[i].input = inputs;
		}
	}

/* time check */
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

	for(int i=0;i<n_carlini;i++){
		if (pthread_create(&networkArray_carlini[i], NULL, (void *(*)(void*))predict_carlini, &net_input_carlini[i]) < 0){
			perror("thread error");
			exit(0);
		}
	}
	for (int i = 0; i < n_carlini; i++){
		pthread_join(networkArray_carlini[i], NULL); // pthread_join : thread 종료를 기다리고 thread 종료 이후 다음 진행
	}

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&timer, start, end);
	std::cout << "Total EXE TIME = "<< timer/1000<<"'s"<< std::endl;

	/* resX time check end */
	sleep(10);
	cudaDeviceSynchronize();
}

                            
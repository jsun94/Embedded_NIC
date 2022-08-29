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
#include "thundersvm/mobile.h"


#include "thundersvm/test.h"

using std::fstream;



#define n_threads 8 // inception의 병렬화 실행을 위한 최소한의 thread 갯수
#define WARMING 4
#define NUMVI 27   /* number of vi_svm */
#define NUMPI 27
#define ALEX_FLATTEN 5

extern void *predict_mobilenet(Net *input);
extern void *predict_mobilenet_warm(Net *input);

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
	int n_mobile=1;
	float time;
	cudaEvent_t start, end;

	int n_all = n_mobile;

	static int stream_index_H = 0;
	static int branch_index_H = 31;


	for(int i=0; i<n_streamPerPool; i++){
	streams.push_back(at::cuda::getStreamFromPool(true,GPU_NUM));
	}

	thpool = thpool_init(n_threads, n_all);

	torch::jit::script::Module mobileModule;

	try {
		mobileModule = torch::jit::load("/home/nvidia/joo/models/MobileNet/mobilenet_jit_cifar10.pt");
		mobileModule.to(device);
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




  /* vi svm path */
	std::string vi_svm_path_list[NUMVI] = {
		"/home/nvidia/joo/models/MobileNet/VISVM/layer4_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer10_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer13_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer16_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer19_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer22_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer25_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer28_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer31_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer34_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer37_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer40_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer43_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer46_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer49_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer52_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer55_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer58_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer61_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer64_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer67_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer70_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer73_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer76_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer79_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer80_visvm.bin",
		"/home/nvidia/joo/models/MobileNet/VISVM/layer81_visvm.bin"
	};
	std::string pi_svm_path_list[NUMPI] = {
		"/home/nvidia/joo/models/MobileNet/PISVM/layer4_layer7_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer7_layer10_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer10_layer13_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer13_layer16_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer16_layer19_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer19_layer22_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer22_layer25_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer25_layer28_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer28_layer31_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer31_layer34_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer34_layer37_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer37_layer40_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer40_layer43_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer43_layer46_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer46_layer49_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer49_layer52_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer52_layer55_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer55_layer58_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer58_layer61_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer61_layer64_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer64_layer67_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer67_layer70_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer70_layer73_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer73_layer76_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer76_layer79_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer79_layer80_pisvm.bin",
		"/home/nvidia/joo/models/MobileNet/PISVM/layer80_layer81_pisvm.bin"
	};
	char* derived_model_path_list[NUMPI+1] = {
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer4_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer7_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer10_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer13_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer16_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer19_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer22_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer25_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer28_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer31_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer34_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer37_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer40_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer43_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer46_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer49_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer52_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer55_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer58_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer61_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer64_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer67_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer70_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer73_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer76_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer79_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer80_derived_model_jit.pt",
		"/home/nvidia/joo/models/MobileNet/D_M/MT/layer81_derived_model_jit.pt"
	};

	/* svm WARM UP input data */
	at::Tensor warm_data = at::zeros({1,1});
	DataSet warm_data_input;
	float warm_label = 0.0;
	warm_data_input.load_from_dense(1, warm_data.sizes()[0], warm_data.data<float>(), &warm_label);
	/* svm WARM UP input data */

	/* derived model WARM UP */
	int dm_warmup_shape[NUMPI+1] = {36992,73984,18496,36992,36992,36992,10368,20736,20736,20736,6400,12800,12800,12800,12800,12800,12800,12800,12800,12800,12800,12800,4608,9216,4096,4096,1024,1024};

  /* vi svm path */

	for (int i = 0; i < NUMVI; i++){
		Visvm v;
		v.index_n = i;
		v.model = vi_load(vi_svm_path_list[i]);
		/* visvm WARMUP */
		v.model->predict_with_core(warm_data_input.instances(),-1, 0, 0);
		vi_svm.push_back(v);
		std::cout << i << " done" << std::endl;
	}
	std::cout << "print : " << vi_svm[0].model->get_n_sv() << std::endl;

  /* derived model path */
//   /* Derived model name list */

	for (int i = 0; i < (NUMPI + 1); i++){
		DM_Pisvm d_p;
		d_p.index_n = i;
		d_p.derived_module = torch::jit::load(derived_model_path_list[i], device);
		// d_p.derived_module.to(device);
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

	torch::Tensor x = torch::ones({1, 3, 32, 32}).to(at::kCUDA); //CIFAR10
	inputs.push_back(x);

	Net net_input_mobile[n_mobile];
	pthread_t networkArray_mobile[n_mobile];



	for(int i=0;i<n_mobile;i++){
		net_input_mobile[i].index_n = i;
		get_submodule_mobilenet(mobileModule, net_input_mobile[i]);
		std::cout << "End get submodule_mobilenet " << i <<"\n";
		net_input_mobile[i].input = inputs;
		net_input_mobile[i].name = "MobileNet";
		// net_input_alex[i].flatten = net_input_alex[i].layers.size() - ALEX_FLATTEN;
		net_input_mobile[i].stream_id = {stream_index_H%n_streamPerPool};
		stream_index_H+=1;
		/*=============WARM UP FOR OPTIMIZATION===============*/
		/*=============FILE===============*/
		//net_input_alex[i].fp = fopen((filename+"-"+"A"+".txt").c_str(),"a");
		for (int j = 0; j < 2; j ++){
			predict_mobilenet_warm(&net_input_mobile[i]);
			net_input_mobile[i].input = inputs;
		}
	}


	std::cout<<"\n==================WARM UP END==================\n";

/* time check */
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);


	for(int i=0;i<n_mobile;i++){
	if (pthread_create(&networkArray_mobile[i], NULL, (void *(*)(void*))predict_mobilenet, &net_input_mobile[i]) < 0){
		perror("thread error");
		exit(0);
	}
	}


	for (int i = 0; i < n_mobile; i++){
	pthread_join(networkArray_mobile[i], NULL);
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

                            
#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <memory>

#include "thundersvm/mobile.h"

using namespace std;
namespace F = torch::nn::functional;
#define NUM_PI 28    //NUMPI + 1
vector<vector<float_type>> vi_result;
vector<vector<float_type>> pi_result;
pthread_cond_t pi_wait_cond[NUM_PI];
pthread_mutex_t pi_wait_mutex[NUM_PI];
int pi_wait_i[NUM_PI]; 


/* VI SVM function */
void *vi_job(Visvm *vi_svm_p){
	/* reshape hidden layer output */
	at::Tensor hidden_out1 = vi_svm_p->hidden_out.detach().to({at::kCPU}).view({-1});

	std::shared_ptr<SvmModel> model = vi_svm_p->model;	/* loaded vi svm */

	DataSet predict_dataset;
	float label = 0.0;
	
	predict_dataset.load_from_dense(1, hidden_out1.sizes()[0], hidden_out1.data<float>(), &label);
	//float_type == double
	vector<float_type> predict_y;

	//skycs
	//predict_y = model->predict(predict_dataset.instances(), -1);
	//gpu == 0   cpu == 1 -> third parameter
	// std::cout << vi_svm_p->index_n << " : " << model->get_n_sv() <<std::endl;
	
	// at::cuda::CUDAStreamGuard guard(streams[7+vi_svm_p->index_n]);
	
	predict_y = model->predict_with_core(predict_dataset.instances(), -1, 0, 0);
	std::cout << vi_svm_p->index_n << " VISVM RESULT : " << predict_y << std::endl;
	pthread_mutex_lock(&vi_mutex_t);
	vi_result.push_back(predict_y);

	vi_cond_i = 0;
	pthread_cond_signal(&vi_cond_t);
	pthread_mutex_unlock(&vi_mutex_t);
}
/* VI SVM function */

/* Derived model & PI SVM function */
// void *pi_job(DM_Pisvm *pi_svm_p){
// 	/* reshape hidden layer output */
// 	at::Tensor hidden_out1 = pi_svm_p->hidden_out.detach().view({1,-1});	/* detach */
// 	std::cout << pi_svm_p->index_n << " " << hidden_out1.sizes() << std::endl;
	
// 	/* Derived model inference */
// 	std::vector<torch::jit::IValue> de_inputs;
// 	de_inputs.push_back(hidden_out1);

// 	// at::cuda::CUDAStreamGuard guard(streams[11+pi_svm_p->index_n]);

// 	at::Tensor de_output = pi_svm_p->derived_module.forward(de_inputs).toTensor();
// 	// at::Tensor de_output = pi_svm_p->derived_module.forward(de_inputs);

// 	// at::Tensor current_out = torch::softmax(de_output, 1).to({at::kCPU}).view({-1});	/* out shape -> (10) */
// 	at::Tensor current_out = de_output.to({at::kCPU}).view({-1});	/* out shape -> (10) */
	
	
// 	//저장
// 	pthread_mutex_lock(&pi_mutex_t);
// 	dm_out_list[pi_svm_p->index_n] = current_out;
// 	dm_out_check[pi_svm_p->index_n] = true;
// 	pthread_mutex_unlock(&pi_mutex_t);

// 	if(pi_svm_p->index_n != (NUM_PI - 1)){
// 		pthread_mutex_lock(&pi_wait_mutex[pi_svm_p->index_n]);
// 		pi_wait_i[pi_svm_p->index_n] = 0;
// 		pthread_cond_signal(&pi_wait_cond[pi_svm_p->index_n]);
// 		pthread_mutex_unlock(&pi_wait_mutex[pi_svm_p->index_n]);
// 	}

// 	if(pi_svm_p->index_n != 0){
// 		while(true){
// 			pthread_mutex_lock(&pi_mutex_t);
// 			bool tf = dm_out_check[pi_svm_p->index_n-1];
// 			pthread_mutex_unlock(&pi_mutex_t);
// 			if(tf){
// 				break;
// 			}
// 			else{
// 				pthread_mutex_lock(&pi_wait_mutex[pi_svm_p->index_n-1]);
// 				while(pi_wait_i[pi_svm_p->index_n-1] == 1){
// 					pthread_cond_wait(&pi_wait_cond[pi_svm_p->index_n-1], &pi_wait_mutex[pi_svm_p->index_n-1]);
// 				}
// 				pi_wait_i[pi_svm_p->index_n-1]=1;
// 				pthread_mutex_unlock(&pi_wait_mutex[pi_svm_p->index_n-1]);
// 			}
// 		}

// 		float* final_out = torch::cat({dm_out_list[pi_svm_p->index_n-1], current_out}).data<float>();

// 		std::shared_ptr<SvmModel> model = pi_svm_p->model;

// 		DataSet predict_dataset;
// 		float label = 0.0;

// 		predict_dataset.load_from_dense(1, 20, final_out, &label);	/*두번째 인자 : (1,20) */
		
// 		vector<float_type> predict_y;
// 		predict_y = model->predict_with_core(predict_dataset.instances(), -1, 1, 1);
// 		std::cout << "PI_SVM - " << pi_svm_p->index_n << ' ' << predict_y << std::endl;
// 		pthread_mutex_lock(&pi_mutex_t);
// 		pi_result.push_back(predict_y);
// 		pi_cond_i = 0;
// 		pthread_cond_signal(&pi_cond_t);
// 		pthread_mutex_unlock(&pi_mutex_t);
// 	}

// }
/* Derived model & PI SVM function */

void get_submodule_mobilenet(torch::jit::script::Module module, Net &net){
	Layer t_layer;
    if(module.children().size() == 0){ 
        t_layer.layer = module;
        net.layers.push_back(t_layer);
        return;
    }
	for(auto children : module.named_children()){
		get_submodule_mobilenet(children.value, net);
	}
}

void *predict_mobilenet_warm(Net *mobile){
	{
		int i;

		for(i=0;i<mobile->layers.size();i++){
			at::Tensor hidden_out = forward_mobilenet(mobile, i);

			mobile->input.clear();
			mobile->input.push_back(hidden_out);
		}


		cudaStreamSynchronize(streams[1]);
	}
}


void *predict_mobilenet(Net *mobile){
	{
		at::cuda::CUDAGuard guard({at::kCUDA,GPU_NUM});
		int i;
		float time;
		cudaEvent_t start, end;

		for(int idx=0; idx<NUM_PI; ++idx){
			pthread_cond_init(&pi_wait_cond[idx], NULL);
			pthread_mutex_init(&pi_wait_mutex[idx], NULL);
			pi_wait_i[idx] = 1;
		}

		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);

		int index_vi = 0;
		int index_pi = 0;
		for(i=0;i<mobile->layers.size();i++){
			at::Tensor hidden_out = forward_mobilenet(mobile, i);

			if (i == 9){
				vi_svm[index_vi].hidden_out = hidden_out; 
				thpool_add_work(thpool,(void(*)(void *))vi_job, &vi_svm[index_vi]);

				index_vi += 1;
			
			}



			mobile->input.clear();
			mobile->input.push_back(hidden_out);
		}
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time, start, end);
		std::cout << "\n*****"<<mobile->name<<" result " <<time/1000<<"s ***** \n";
		std::cout << (mobile->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
		cudaStreamSynchronize(streams[1]);

		while(true){
			pthread_mutex_lock(&vi_mutex_t);
			int size = vi_result.size();
			pthread_mutex_unlock(&vi_mutex_t);
			if(size == NUMVI){
			// cudaStreamSynchronize(streams[7]);
			// cudaStreamSynchronize(streams[8]);
			// cudaStreamSynchronize(streams[9]);
			// cudaStreamSynchronize(streams[10]);
				break;
			}
			else{
				pthread_mutex_lock(&vi_mutex_t);
				while(vi_cond_i == 1){
					pthread_cond_wait(&vi_cond_t, &vi_mutex_t);
				}
				vi_cond_i = 1;
				pthread_mutex_unlock(&vi_mutex_t);
			}
		}

		// while(true){
		// 	pthread_mutex_lock(&pi_mutex_t);
		// 	int size = pi_result.size();
		// 	pthread_mutex_unlock(&pi_mutex_t);
		// 	if(size == (NUM_PI - 1)){
		// 	// cudaStreamSynchronize(streams[11]);
		// 	// cudaStreamSynchronize(streams[12]);
		// 	// cudaStreamSynchronize(streams[13]);
		// 	// cudaStreamSynchronize(streams[14]);
		// 		break;
		// 	}
		// 	else{
		// 		pthread_mutex_lock(&pi_mutex_t);
		// 		while(pi_cond_i == 1){
		// 			pthread_cond_wait(&pi_cond_t, &pi_mutex_t);
		// 		}
		// 		pi_cond_i = 1;
		// 		pthread_mutex_unlock(&pi_mutex_t);
		// 	}
		// }

	}
}

at::Tensor forward_mobilenet(Net * net, int idx){
	std::vector<torch::jit::IValue> inputs = net->input;
	at::Tensor out;
	{
		at::cuda::CUDAStreamGuard guard(streams[1]); // high, low
		out = net->layers[idx].layer.forward(inputs).toTensor();
		net->layers[idx].output = out;
	}
	return out;	
}


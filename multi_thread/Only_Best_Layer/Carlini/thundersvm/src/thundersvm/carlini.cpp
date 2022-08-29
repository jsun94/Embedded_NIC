#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <memory>
#include "cuda_runtime.h"
#include "thundersvm/carlini.h"
#include <typeinfo>
#include <fstream>
#include <time.h>
namespace F = torch::nn::functional;
#define NUM_PI 9    //NUMPI + 1
// #define NUM_PI 5

vector<vector<float_type>> vi_result;
vector<vector<float_type>> pi_result;
pthread_cond_t pi_wait_cond[NUM_PI];
pthread_mutex_t pi_wait_mutex[NUM_PI];
int pi_wait_i[NUM_PI];

/* VI SVM function */
void *vi_job(Visvm *vi_svm_p){
	/* reshape hidden layer output */

	at::Tensor hidden_out1 = vi_svm_p->hidden_out.detach().to({at::kCPU}).view({-1});
	// at::Tensor hidden_out1 = vi_svm_p->hidden_out.detach().to({at::kCPU}).view({5000,-1});



	std::shared_ptr<SvmModel> model = vi_svm_p->model;	/* loaded vi svm */


	DataSet predict_dataset;
	float label = 0.0;

	
	predict_dataset.load_from_dense(1, hidden_out1.sizes()[0], hidden_out1.data<float>(), &label);


	//float_type == double
	vector<float_type> predict_y;

	//skycs
	//predict_y = model->predict(predict_dataset.instances(), -1);
	//gpu == 0   cpu == 1 -> third parameter

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
void *pi_job(DM_Pisvm *pi_svm_p){
	/* reshape hidden layer output */
	// at::Tensor hidden_out1 = pi_svm_p->hidden_out.detach().view({1,-1});	/* detach */
	at::Tensor hidden_out1 = pi_svm_p->hidden_out.detach().view({5000,-1});	/* detach */

	/* Derived model inference */
	std::vector<torch::jit::IValue> de_inputs;
	de_inputs.push_back(hidden_out1);
	// at::cuda::CUDAStreamGuard guard(streams[11+pi_svm_p->index_n]);
	at::Tensor de_output = pi_svm_p->derived_module.forward(de_inputs).toTensor();
	std::cout << "PI NUM : " << pi_svm_p->index_n <<" "<<hidden_out1.sizes() <<std::endl;
	// at::Tensor current_out = torch::softmax(de_output, 1).to({at::kCPU}).view({-1});	/* out shape -> (10) */
	at::Tensor current_out = de_output.to({at::kCPU}).view({-1});	/* out shape -> (10) */
	//저장
	pthread_mutex_lock(&pi_mutex_t);
	dm_out_list[pi_svm_p->index_n] = current_out;
	dm_out_check[pi_svm_p->index_n] = true;
	pthread_mutex_unlock(&pi_mutex_t);

	if(pi_svm_p->index_n != (NUM_PI-1)){
		pthread_mutex_lock(&pi_wait_mutex[pi_svm_p->index_n]);
		pi_wait_i[pi_svm_p->index_n] = 0;
		pthread_cond_signal(&pi_wait_cond[pi_svm_p->index_n]);
		pthread_mutex_unlock(&pi_wait_mutex[pi_svm_p->index_n]);
	}

	if(pi_svm_p->index_n != 0){
		while(true){
			pthread_mutex_lock(&pi_mutex_t);
			bool tf = dm_out_check[pi_svm_p->index_n-1];
			pthread_mutex_unlock(&pi_mutex_t);
			if(tf){
				break;
			}
			else{
				pthread_mutex_lock(&pi_wait_mutex[pi_svm_p->index_n-1]);
				while(pi_wait_i[pi_svm_p->index_n-1] == 1){
					pthread_cond_wait(&pi_wait_cond[pi_svm_p->index_n-1], &pi_wait_mutex[pi_svm_p->index_n-1]);
				}
				pi_wait_i[pi_svm_p->index_n-1]=1;
				pthread_mutex_unlock(&pi_wait_mutex[pi_svm_p->index_n-1]);
			}
		}

		float* final_out = torch::cat({dm_out_list[pi_svm_p->index_n-1], current_out}).data<float>();

		std::shared_ptr<SvmModel> model = pi_svm_p->model;

		DataSet predict_dataset;
		float label = 0.0;

		predict_dataset.load_from_dense(1, 20, final_out, &label);	/*두번째 인자 : (1,20) */
		
		vector<float_type> predict_y;
		predict_y = model->predict_with_core(predict_dataset.instances(), -1, 1, 1);
		std::cout << "PI_SVM - " << pi_svm_p->index_n << ' ' << std::endl;
		pthread_mutex_lock(&pi_mutex_t);
		pi_result.push_back(predict_y);
		pi_cond_i = 0;
		pthread_cond_signal(&pi_cond_t);
		pthread_mutex_unlock(&pi_mutex_t);
	std::cout << "PI NUM : " << pi_svm_p->index_n <<" done" <<std::endl;
	}

}
/* Derived model & PI SVM function */

void get_submodule_carlini(torch::jit::script::Module module, Net &net){
	Layer t_layer;
    if(module.children().size() == 0){ 
        t_layer.layer = module;
        net.layers.push_back(t_layer);
        return;
    }
	for(auto children : module.named_children()){
		get_submodule_carlini(children.value, net);
	}
}

void *predict_carlini_warm(Net *carlini){
	std::vector<torch::jit::IValue> inputs = carlini->input;
	int i;	

	int index_help = 0;
	for(i=0;i<carlini->layers.size();i++){
		at::Tensor hidden_out = forward_carlini(carlini, i);
		carlini->input.clear();
		carlini->input.push_back(hidden_out);
	}
	
	cudaStreamSynchronize(streams[carlini->index_n % n_streamPerPool]);
}


void *predict_carlini(Net *carlini){
	std::vector<torch::jit::IValue> inputs = carlini->input;
	int i;
	float time_r;
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
	for(i=0;i<carlini->layers.size();i++){
		at::Tensor hidden_out = forward_carlini(carlini, i);

		if(i == 4){
		// 여기서 VI job을 thread_pool로 보냄
			vi_svm[index_vi].hidden_out = hidden_out; 
			thpool_add_work(thpool,(void(*)(void *))vi_job, &vi_svm[index_vi]);
			index_vi += 1;
		}

		// th_arg th;
		carlini->input.clear();
		carlini->input.push_back(hidden_out);
	}
	cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_r, start, end);
	std::cout << "\n*****"<<carlini->name<<" result*****" << "     Carlini exe time >>> " << time_r/1000 << "'s" <<std::endl;
	// std::cout << (carlini->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	std::cout << "carlini layers typeid : " << typeid(carlini->layers[1].output).name() << "\n";
	// std::cout << "layers output size : " << carlini->layers[2].output.sizes() << "\n";
	std::cout << "layers output size : " << carlini->layers[14].output.sizes() << "\n";
	cudaStreamSynchronize(streams[carlini->index_n % n_streamPerPool]);
	
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
	// 	if(size == NUM_PI-1){
	// 		// cudaStreamSynchronize(streams[11]);
	// 		// cudaStreamSynchronize(streams[12]);
	// 		// cudaStreamSynchronize(streams[13]);
	// 		// cudaStreamSynchronize(streams[14]);
	// 		// cudaStreamSynchronize(streams[15]);
	// 		// cudaStreamSynchronize(streams[16]);
	// 		// cudaStreamSynchronize(streams[17]);
	// 		// cudaStreamSynchronize(streams[18]);

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


at::Tensor forward_carlini(Net* net, int idx){
	std::vector<torch::jit::IValue> inputs = net->input;
	at::Tensor out;
	{
		at::cuda::CUDAStreamGuard guard(streams[(net->index_n)%n_streamPerPool]); // high, low
		out = net->layers[idx].layer.forward(inputs).toTensor();
		net->layers[idx].output = out;
	}
	return out;		
}
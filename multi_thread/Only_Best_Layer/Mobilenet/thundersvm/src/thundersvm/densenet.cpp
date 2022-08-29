// #include <torch/script.h>
// #include <torch/torch.h>
// #include <typeinfo>
// #include <iostream>
// #include <inttypes.h>
// #include <memory>
// #include <time.h>
// #include "cuda_runtime.h"

// #include "thundersvm/densenet.h"


// using namespace std;
// namespace F = torch::nn::functional;
// #define NUM_PI 8
// vector<vector<float_type>> vi_result;
// vector<vector<float_type>> pi_result;
// pthread_cond_t pi_wait_cond[NUM_PI];
// pthread_mutex_t pi_wait_mutex[NUM_PI];
// int pi_wait_i[NUM_PI]; 

// /* VI SVM function */
// void *vi_job(Visvm *vi_svm_p){
// 	/* reshape hidden layer output */
// 	at::Tensor hidden_out1 = vi_svm_p->hidden_out.detach().to({at::kCPU}).view({-1});

// 	std::shared_ptr<SvmModel> model = vi_svm_p->model;	/* loaded vi svm */

// 	DataSet predict_dataset;
// 	float label = 0.0;
	
// 	predict_dataset.load_from_dense(1, hidden_out1.sizes()[0], hidden_out1.data<float>(), &label);
// 	//float_type == double
// 	vector<float_type> predict_y;

// 	//skycs
// 	//predict_y = model->predict(predict_dataset.instances(), -1);
// 	//gpu == 0   cpu == 1 -> third parameter
// 	// std::cout << vi_svm_p->index_n << " : " << model->get_n_sv() <<std::endl;
// 	predict_y = model->predict_with_core(predict_dataset.instances(), -1, 0, 0);
// 	std::cout << vi_svm_p->index_n << " VISVM RESULT : " << predict_y << std::endl;
// 	pthread_mutex_lock(&vi_mutex_t);
// 	vi_result.push_back(predict_y);

// 	vi_cond_i = 0;
// 	pthread_cond_signal(&vi_cond_t);
// 	pthread_mutex_unlock(&vi_mutex_t);
// }
// /* VI SVM function */

// /* Derived model & PI SVM function */
// void *pi_job(DM_Pisvm *pi_svm_p){
// 	/* reshape hidden layer output */
// 	std::cout << "here" << std::endl;
// 	at::Tensor hidden_out1 = pi_svm_p->hidden_out.detach().view({1,-1});	/* detach */
// 	std::cout << pi_svm_p->index_n << " " << hidden_out1.sizes() << std::endl;
// 	/* Derived model inference */
// 	std::vector<torch::jit::IValue> de_inputs;
// 	de_inputs.push_back(hidden_out1);
// 	std::cout << "here2" << std::endl;

// 	at::Tensor de_output = pi_svm_p->derived_module.forward(de_inputs).toTensor();
// 	std::cout << "here3" << std::endl;

// 	at::Tensor current_out = torch::softmax(de_output, 1).to({at::kCPU}).view({-1});	/* out shape -> (10) */
// 	//저장
// 	pthread_mutex_lock(&pi_mutex_t);
// 	dm_out_list[pi_svm_p->index_n] = current_out;
// 	dm_out_check[pi_svm_p->index_n] = true;
// 	pthread_mutex_unlock(&pi_mutex_t);

// 	if(pi_svm_p->index_n != 8){
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
// 		//std::cout << "PI_SVM - " << pi_svm_p->index_n << ' ' << std::endl;
// 		pthread_mutex_lock(&pi_mutex_t);
// 		pi_result.push_back(predict_y);
// 		pi_cond_i = 0;
// 		pthread_cond_signal(&pi_cond_t);
// 		pthread_mutex_unlock(&pi_mutex_t);
// 	}

// }
// /* Derived model & PI SVM function */


// void get_submodule_densenet(torch::jit::script::Module module,Net &net){
// 	Dummy concat;
//     Layer t_layer;
//     if(module.children().size() == 0){
//         t_layer.layer = module;
//         t_layer.name = "Normal";
//         net.layers.push_back(t_layer);
//         return;
//     }
//     for(auto children : module.named_children()){
//         //DenseBlock - is configured with multiple denselayer 
//         if(children.name.find("denseblock") != std::string::npos){
//             int size = net.layers.size();
//             for(auto layer : children.value.named_children()){
//                 //Denselayer - is configured with six layer modules
//                 if(layer.name.find("denselayer") != std::string::npos){
//                     t_layer.from_idx = {-1};
//                     t_layer.layer = concat;
//                     t_layer.name = "concat";
//                     net.layers.push_back(t_layer);
//                     for(auto in_denselayer : layer.value.named_children()){
//                         t_layer.from_idx.clear();
//                         t_layer.layer = in_denselayer.value;
//                         t_layer.name = "Normal";
//                         net.layers.push_back(t_layer);
                        
//                     }
//                     t_layer.from_idx = {-7, -1};
//                     t_layer.layer = concat;
//                     t_layer.name = "concat";
//                     net.layers.push_back(t_layer);
//                 }
//                 else
//                     get_submodule_densenet(layer.value, net);
//             }
//             continue;
//         }
//         //Transition
//         get_submodule_densenet(children.value, net);
//     }
// }

// void *predict_densenet(Net *densenet){
//     int i;
//     float time;
//     cudaEvent_t start, end;
//     cudaEventCreate(&start);
//     cudaEventCreate(&end);
//     cudaEventRecord(start);

// 	int index_help = 0;
//     for(i=0;i<densenet->layers.size();i++){
// 		cond_i[densenet->index_n] = 1; 
//         netlayer nl;
// 		nl.net = densenet;
//         nl.net->index = i;

// 		th_arg th;
// 		th.arg = &nl;

//         // thpool_add_work(thpool,(void(*)(void *))forward_densenet,(void*) &th);
// 		at::Tensor hidden_out = forward_densenet(densenet, i);

// 		if(i < 9){
// 			vi_svm[index_help].hidden_out = hidden_out; 
// 			thpool_add_work(thpool,(void(*)(void *))vi_job, &vi_svm[index_help]);

// 			// pi_svm[index_help].hidden_out = hidden_out;
// 			// thpool_add_work(thpool,(void(*)(void *))pi_job,&pi_svm[index_help]);

// 			index_help += 1;			
// 		}


// 		densenet->input.clear();
// 		// densenet->input.push_back(densenet->layers[i].output);
// 		densenet->input.push_back(hidden_out);
        
//     }

// 	while(true){
// 		pthread_mutex_lock(&vi_mutex_t);
// 		int size = vi_result.size();
// 		pthread_mutex_unlock(&vi_mutex_t);
// 		if(size == 9){
// 			break;
// 		}
// 		else{
// 			pthread_mutex_lock(&vi_mutex_t);
// 			while(vi_cond_i == 1){
// 				pthread_cond_wait(&vi_cond_t, &vi_mutex_t);
// 			}
// 			vi_cond_i = 1;
// 			pthread_mutex_unlock(&vi_mutex_t);
// 		}
// 	}
// 	// while(true){
// 	// 	pthread_mutex_lock(&pi_mutex_t);
// 	// 	int size = pi_result.size();
// 	// 	pthread_mutex_unlock(&pi_mutex_t);
// 	// 	if(size == 8){
// 	// 		break;
// 	// 	}
// 	// 	else{
// 	// 		pthread_mutex_lock(&pi_mutex_t);
// 	// 		while(pi_cond_i == 1){
// 	// 			pthread_cond_wait(&pi_cond_t, &pi_mutex_t);
// 	// 		}
// 	// 		pi_cond_i = 1;
// 	// 		pthread_mutex_unlock(&pi_mutex_t);
// 	// 	}
// 	// }

//     //cudaEventCreate(&event_A);
//     cudaStreamSynchronize(streams[densenet->H_L][(densenet->index_s)%n_streamPerPool]);
//     cudaEventRecord(end);
//     cudaEventSynchronize(end);
//     cudaEventElapsedTime(&time, start, end);
// 	//std::cout<< "Dense model end time is" << exe_time << std::endl;
//     std::cout << "\n*****"<<densenet->name<<" result*****" << "     Densenet exe time >>> " << time/1000 << "'s" <<std::endl;
// 	std::cout << "index num = "<< densenet->index_n << "	priority num = "<< densenet->priority << "     Stream [" << densenet->H_L << "][" << (densenet->index_s)%n_streamPerPool << "]" << std::endl;
// 	std::cout << (densenet->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
//     std::cout << " " << std::endl;
// }

// at::Tensor forward_densenet(Net *net, int idx){ 
//     pthread_mutex_lock(&mutex_t[net->index_n]);
// 	std::vector<torch::jit::IValue> inputs = net->input;
//     int k =net->index;
//     at::Tensor out;

//     //at::cuda::setCurrentCUDAStream(streams[(nl->net->index_n)]);
//     {   
//         at::cuda::CUDAStreamGuard guard(streams[net->H_L][(net->index_s)%n_streamPerPool]); // high, low
// 		// int p;
// 		// cudaStreamGetindex_n(streams[(nl->net->index_n%n_streamPerPool)],&p);
// 		// std::cout <<"index "<< (nl->net->index_n%n_streamPerPool) << "dense index_n num" << p << std::endl;
//         //at::cuda::CUDAStream p();
//         //std::cout<<"Dense stream id = "<<streams[0].unwrap().id()<<"\n";
        
//         //std::cout << "Dense stream index : "<<((nl->net->index_n)%n_streamPerPool) << std::endl;
//         if(k == net->flatten){
//             out = F::relu(inputs[0].toTensor(), F::ReLUFuncOptions().inplace(true));
//             out = F::adaptive_avg_pool2d(out, F::AdaptiveAvgPool2dFuncOptions(1));
//             out = out.view({out.size(0), -1});
//             inputs.clear();
//             inputs.push_back(out);
//             out = net->layers[k].layer.forward(inputs).toTensor();
//         }
//         else if(net->layers[k].name == "concat"){
//             std::vector<at::Tensor> cat_input;
//             for(int i=0;i<net->layers[k].from_idx.size();i++){
//                 cat_input.push_back(net->layers[k + net->layers[k].from_idx[i]].output);
//             }
//             out = torch::cat(cat_input, 1);
//         }
//         else{
//             out = net->layers[k].layer.forward(inputs).toTensor();
//         }
//     }
//     net->layers[k].output = out;
// 	cond_i[net->index_n]=0;
// 	pthread_cond_signal(&cond_t[net->index_n]);
// 	pthread_mutex_unlock(&mutex_t[net->index_n]);
// 	return out;
// }
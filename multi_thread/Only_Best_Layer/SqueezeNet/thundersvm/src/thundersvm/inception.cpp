// #include <torch/script.h>
// #include <torch/torch.h>
// #include <typeinfo>
// #include <iostream>
// #include <inttypes.h>
// #include <functional>
// #include <memory>
// #include <thread>
// #include <unistd.h>
// #include "thundersvm/inception.h"
// #include "cuda_runtime.h"
// #include <time.h>
// #include <math.h>
// /*
// event_idx : branch_num in inception (for recording event)
// input_idx : the index of the input from the current layer
// skip : Number of layer modules in one branch (How many more signals do thread have to send)
// branch_idx : The last layer index of the branch to determine if the operation is complete(exe_success)
// */

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

// void get_submodule_inception(torch::jit::script::Module module, Net &net){
//     Layer t_layer;    
//     Dummy temp;
//     for(auto children : module.named_children()){
//         if(children.name == "Mixed_5b" || children.name == "Mixed_5c" || children.name == "Mixed_5d"){ //InceptionA
//             int event_idx = -1;
//             for(auto branch : children.value.named_children()){
//                 if(branch.name == "branch_pool"){
//                     t_layer.layer = temp;
//                     t_layer.exe_success = false;
//                     t_layer.input_idx = -7;
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.name = "avg_pool2d";
//                     t_layer.skip = 2;
//                     net.layers.push_back(t_layer);    
//                     t_layer.input_idx = 0;
//                     t_layer.skip = 0;
//                     t_layer.branch_idx = {-7, -5, -2, 0};
//                 }
//                 if(branch.name == "branch1x1"){
//                     t_layer.input_idx = 0;
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.skip = 1;
//                     t_layer.branch_idx = {2, 5, 7};
//                 }
//                 else if(branch.name == "branch5x5_1"){
//                     t_layer.input_idx = -2;
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.skip = 2;
//                 }
//                 else if(branch.name == "branch5x5_2"){
//                     t_layer.input_idx = 0;
//                     t_layer.skip = 0;
//                     t_layer.branch_idx = {-2, 3, 5};
//                 }
//                 else if(branch.name == "branch3x3dbl_1"){
//                     t_layer.input_idx = -4;
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.skip = 3;
//                 }
//                 else if(branch.name == "branch3x3dbl_3"){
//                     t_layer.input_idx = 0;
//                     t_layer.skip = 0;
//                     t_layer.branch_idx = {-5, -3, 2};
//                 }
//                 else{
//                     t_layer.input_idx = 0;
//                     t_layer.skip = 0;
//                 }
//                 t_layer.name = "A_" + branch.name;
//                 t_layer.layer = branch.value;
//                 t_layer.exe_success = false;
//                 net.layers.push_back(t_layer);
//             }
//             t_layer.event_idx = -1;
//             t_layer.input_idx = 0;
//             t_layer.from_idx = {-8,-6,-3, -1};
//             t_layer.layer = temp;
//             t_layer.exe_success = false;
//             t_layer.name = "concat";
//             t_layer.skip = 0;
//             net.layers.push_back(t_layer);
//             continue;
//         }
//         else if(children.name == "Mixed_6a"){   //InceptionB
//             int event_idx = -1;
//             for(auto branch : children.value.named_children()){
//                 if(branch.name == "branch3x3"){
//                     t_layer.input_idx = 0;
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.skip = 1;
//                     t_layer.branch_idx = {3, 4};
//                 }
//                 else if(branch.name == "branch3x3dbl_1"){
//                     t_layer.input_idx = -2;
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.skip = 3;
//                 }
//                 else if(branch.name == "branch3x3dbl_3"){
//                     t_layer.input_idx = 0;
//                     t_layer.skip = 0;
//                     t_layer.branch_idx = {-3, 1};
//                 }
//                 else{
//                     t_layer.input_idx = 0;
//                     t_layer.skip = 0;
//                 }
//                 t_layer.layer = branch.value;
//                 t_layer.exe_success = false;
//                 t_layer.name = "B_" + branch.name;
//                 net.layers.push_back(t_layer);
//                 if(branch.name == "branch3x3dbl_3"){
//                     t_layer.input_idx = -5;
//                     t_layer.layer = temp;
//                     t_layer.exe_success = false;
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.name = "max_pool2d";
//                     t_layer.skip = 1;
//                     t_layer.branch_idx = {-4, -1, 0};
//                     net.layers.push_back(t_layer);
//                 }
//             }
//             t_layer.event_idx = -1;
//             t_layer.input_idx = 0;
//             t_layer.from_idx = {-5,-2, -1};
//             t_layer.layer = temp;
//             t_layer.exe_success = false;
//             t_layer.name = "concat";
//             t_layer.skip = 0;
//             net.layers.push_back(t_layer);
//             continue;
//         }
//         else if(children.name == "Mixed_6b" || children.name == "Mixed_6c" || children.name == "Mixed_6d" || children.name == "Mixed_6e" ){ //InceptionC
//             int event_idx = -1;
//             for(auto branch : children.value.named_children()){
//                 if(branch.name == "branch_pool"){
//                     t_layer.input_idx = -10;
//                     t_layer.layer = temp;
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.exe_success = false;
//                     t_layer.name = "avg_pool2d";
//                     t_layer.skip = 2;
//                     net.layers.push_back(t_layer);
//                     t_layer.input_idx = 0;
//                     t_layer.skip = 0;
//                     t_layer.branch_idx = {-10,-7,-2, 0};
//                 }
//                 else if(branch.name == "branch1x1"){
//                     t_layer.input_idx = 0;
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.skip = 1;
//                     t_layer.branch_idx = {3,8,10};
//                 }
//                 else if(branch.name == "branch7x7_1"){
//                     t_layer.input_idx = -2;
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.skip = 3;
//                 }
//                 else if(branch.name == "branch7x7_3"){
//                     t_layer.input_idx = 0;
//                     t_layer.skip = 0;
//                     t_layer.branch_idx = {-3,5,7};
//                 }
//                 else if(branch.name == "branch7x7dbl_1"){
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.input_idx = -5;
//                     t_layer.skip = 5;
//                 }
//                 else if(branch.name == "branch7x7dbl_3"){
//                     t_layer.input_idx = 0;
//                     t_layer.skip = 0;
//                     t_layer.branch_idx = {-8,-5,2};
//                 }
//                 else{
//                     t_layer.skip = 0;
//                     t_layer.input_idx = 0;
//                 }
//                 t_layer.layer = branch.value;
//                 t_layer.exe_success = false;
//                 t_layer.name = "C_" + branch.name;
//                 net.layers.push_back(t_layer);
//             }
//             t_layer.event_idx = -1;
//             t_layer.from_idx = {-11,-8,-3, -1};
//             t_layer.layer = temp;
//             t_layer.exe_success = false;
//             t_layer.name = "concat";
//             t_layer.skip = 0;
//             net.layers.push_back(t_layer);
//             continue;
//         }
//         else if(children.name == "Mixed_7a"){   //InceptionD
//             int event_idx = -1;
//             for(auto branch : children.value.named_children()){
//                 t_layer.skip = 0;
//                 if(branch.name == "branch7x7x3_1"){
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.input_idx = -3;
//                     t_layer.skip = 4;
//                 }
//                 else {
//                     t_layer.input_idx = 0;
//                     if(branch.name == "branch3x3_1"){
//                         t_layer.skip = 2;
//                         t_layer.event_idx = ++event_idx;
//                     }
//                     else if(branch.name == "branch7x7x3_4"){
//                         t_layer.branch_idx = {-4, 1};
//                     }
//                     else if(branch.name == "branch3x3_2"){
//                         t_layer.branch_idx = {4, 5};
//                     }
//                 }
//                 t_layer.layer = branch.value;
//                 t_layer.exe_success = false;
//                 t_layer.name = "D_" + branch.name;
//                 net.layers.push_back(t_layer);
//                 if(branch.name == "branch7x7x3_4"){
//                     t_layer.input_idx = -7;
//                     t_layer.layer = temp;
//                     t_layer.skip = 1;
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.exe_success = false;
//                     t_layer.name = "max_pool2d";
//                     t_layer.branch_idx = {-5, -1, 0};
//                     net.layers.push_back(t_layer);
//                 }
//             }
//             t_layer.event_idx = -1;
//             t_layer.from_idx = {-6,-2, -1};
//             t_layer.layer = temp;
//             t_layer.exe_success = false;
//             t_layer.skip = 0;
//             t_layer.name = "concat";
//             net.layers.push_back(t_layer);
//             continue;
//         }
//         else if(children.name == "Mixed_7b" || children.name == "Mixed_7c"){    //InceptionE
//             int event_idx = -1;
//             for(auto branch : children.value.named_children()){
//                 t_layer.skip = 0;
//                 if(branch.name == "branch_pool"){
//                     t_layer.input_idx = -11;
//                     t_layer.layer = temp;
//                     t_layer.exe_success = false;
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.name = "avg_pool2d";
// 	                t_layer.skip = 2;
//                     net.layers.push_back(t_layer);
//                     t_layer.branch_idx = {-11, -7, -2, 0}; 
//                     t_layer.input_idx = 0;
//                 }
//                 else if(branch.name == "branch3x3_1" || branch.name == "branch3x3_2b" || branch.name == "branch3x3dbl_3b"){
//                     t_layer.input_idx = -2;
//                     if(branch.name == "branch3x3_1"){
// 	                    t_layer.skip = 4;
//                         t_layer.event_idx = ++event_idx;
//                     }
//                 }
//                 else if(branch.name == "branch3x3dbl_1"){
//                     t_layer.event_idx = ++event_idx;
// 	                t_layer.skip = 5;
//                     t_layer.input_idx = -6;
//                 }
//                 else{
//                     t_layer.input_idx = 0;
//                     if(branch.name == "branch1x1"){
//                         t_layer.skip = 1;
//                         t_layer.event_idx = ++event_idx;
//                         t_layer.branch_idx = {4, 9, 11};
//                     }
//                 }
//                 t_layer.layer = branch.value;
//                 t_layer.exe_success = false;
//                 t_layer.name = "E_" + branch.name;
//                 net.layers.push_back(t_layer);
//                 if(branch.name == "branch3x3_2b" || branch.name == "branch3x3dbl_3b"){
//                     if(branch.name == "branch3x3dbl_3b") t_layer.branch_idx = {-9, -5, 2}; 
//                     else t_layer.branch_idx = {-4, 5, 7}; 
//                     t_layer.input_idx = 0;
//                     t_layer.from_idx = {-2, -1};
//                     t_layer.layer = temp;
//                     t_layer.skip = 0;
//                     t_layer.exe_success = false;
//                     t_layer.name = "concat";
//                     net.layers.push_back(t_layer);
//                 }
//             }
//             t_layer.skip = 0;
//             t_layer.input_idx = 0;
//             t_layer.from_idx = {-12,-8,-3, -1};
//             t_layer.layer = temp;
//             t_layer.exe_success = false;
//             t_layer.event_idx =-1;
//             t_layer.name = "concat";
//             net.layers.push_back(t_layer);
//             continue;
//         }
//         else if(children.name != "AuxLogits")
//         {   
//             t_layer.input_idx = 0;
//             t_layer.event_idx = -1;
//             t_layer.layer = children.value;
//             t_layer.skip = 0;
//             t_layer.name = children.name;
//             t_layer.exe_success = false;
//             net.layers.push_back(t_layer);   
//         }
//     }
// }


// void *predict_inception(Net *inception){
	
// 	int i;
    
//     auto x_ch0 = torch::unsqueeze(inception->input[0].toTensor().index({torch::indexing::Slice(), 0}), 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5;
//     auto x_ch1 = torch::unsqueeze(inception->input[0].toTensor().index({torch::indexing::Slice(), 1}), 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5;
//     auto x_ch2 = torch::unsqueeze(inception->input[0].toTensor().index({torch::indexing::Slice(), 2}), 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5;
    
//     x_ch0.to(at::kCUDA);
//     x_ch1.to(at::kCUDA);
//     x_ch2.to(at::kCUDA);

//     auto x_cat = torch::cat({x_ch0,x_ch1,x_ch2},1).to(at::kCUDA);
//     // std::vector<int> stream_id = {(inception->index_n)%n_streamPerPool, abs(n_streamPerPool-((inception->index_n)*3+1))%n_streamPerPool, abs(n_streamPerPool-((inception->index_n)*3+2))%n_streamPerPool, abs(n_streamPerPool-((inception->index_n)*3+3))%n_streamPerPool};
//     std::vector<int> stream_id = {(inception->index_s)%n_streamPerPool, abs(inception->index_b)%n_streamPerPool, abs((inception->index_b)-1)%n_streamPerPool, abs((inception->index_b)-2)%n_streamPerPool};
//     inception->input[0] = x_cat;

// 	for(int idx=0; idx<NUM_PI; ++idx){
// 		pthread_cond_init(&pi_wait_cond[idx], NULL);
// 		pthread_mutex_init(&pi_wait_mutex[idx], NULL);
// 		pi_wait_i[idx] = 1;
// 	}

//     float time;
//     cudaEvent_t start, end;
//     cudaEventCreate(&start);
//     cudaEventCreate(&end);
//     cudaEventRecord(start);
    
// 	int index_help = 0;    
// 	for(i=0;i<inception->layers.size();i++){        
// 		cond_i[inception->index_n] = 1;
// 		inception->layers[i].exe_success = false;

// 		netlayer nl;
// 		nl.net = inception;
// 		nl.net->index = i;

// 		th_arg th;
// 		th.arg = &nl;

// 		// std::cout << "layer index : "<< inception->index << " name : " << inception->layers[i].name << std::endl;
// 		at::Tensor hidden_out = forward_inception(inception, i);
// 		if(i < 9){
// 			vi_svm[index_help].hidden_out = hidden_out; 
// 			thpool_add_work(thpool,(void(*)(void *))vi_job, &vi_svm[index_help]);

// 			// pi_svm[index_help].hidden_out = hidden_out;
// 			// thpool_add_work(thpool,(void(*)(void *))pi_job,&pi_svm[index_help]);

// 			index_help += 1;			
// 		}
 
//         i = nl.net->index;
// 		inception->input.clear();

// 		// inception->input.push_back(inception->layers[i].output);
// 		inception->input.push_back(hidden_out);

// 		// pthread_mutex_unlock(&mutex_t[inception->index_n]);
//         cudaStreamSynchronize(streams[inception->H_L][stream_id[0]]);
// 	}

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

//     //cudaStreamSynchronize(streams[inception->H_L][stream_id[0]]);
//     cudaEventRecord(end);
//     cudaEventSynchronize(end);
//     cudaEventElapsedTime(&time, start, end);

//     std::cout << "\n*****"<<inception->name<<" result*****" << "     Inception v3 exe time >>> " << time/1000 << "'s" <<std::endl;
// 	std::cout << "index num = "<< inception->index_n << "	priority num = "<< inception->priority << "     Stream [" << inception->H_L << "][" << stream_id[0] << "]" << std::endl;
//     std::cout << "Branch Stream ["<<stream_id[1]<<"] ["<<stream_id[2]<<"] ["<<stream_id[3]<<"]" << std::endl;
// 	std::cout << (inception->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
//     printf("\n");
//     }

// at::Tensor forward_inception(Net *net, int idx){
// 	pthread_mutex_lock(&mutex_t[net->index_n]);
	
// 	int k = net->index;
//     int n_all = net->n_all;
//     std::vector<torch::jit::IValue> inputs;
//     //std::vector<int> stream_id = {(nl->net->index_n)%n_streamPerPool, abs(n_streamPerPool-((nl->net->index_n)*3+1))%n_streamPerPool, abs(n_streamPerPool-((nl->net->index_n)*3+2))%n_streamPerPool, abs(n_streamPerPool-((nl->net->index_n)*3+3))%n_streamPerPool};
//     std::vector<int> stream_id = {(net->index_s)%n_streamPerPool, abs(net->index_b)%n_streamPerPool, abs((net->index_b)-1)%n_streamPerPool, abs((net->index_b)-2)%n_streamPerPool};
//     if(net->layers[k].input_idx != 0){
//         inputs.push_back(net->layers[k + net->layers[k].input_idx].output);
//     }
//     else {
//         inputs = net->input;
//     }

// 	at::Tensor out;
//     {
//         at::cuda::CUDAStreamGuard guard(streams[net->H_L][stream_id[0]]);
//         if(k == net->flatten){
//             out = inputs[0].toTensor().view({inputs[0].toTensor().size(0), -1});
//             inputs.clear();
//             inputs.push_back(out);
//             out = net->layers[k].layer.forward(inputs).toTensor();
//         }
//         else if(net->layers[k].skip > 0){   //branch
//             //at::cuda::setCurrentCUDAStream(streams[stream_id[(nl->net->layers[k].event_idx)%4]]);
//             {
//                 at::cuda::CUDAStreamGuard guard(streams[net->H_L][stream_id[(net->layers[k].event_idx)%4]]); //event_idx == branch_num
//                 out = inputs[0].toTensor();
//                 int T = net->layers[k].skip;
//                 for(int t=0;t<T;k++,t++){
//                     if(net->layers[k].input_idx != 0){
//                         inputs.clear();
//                         inputs.push_back(net->layers[k + net->layers[k].input_idx].output);
//                     }
//                     else {
//                         inputs.clear();
//                         inputs.push_back(out);
//                     } 
                    
//                     if(net->layers[k].name == "concat"){
//                         std::vector<at::Tensor> cat_input;
//                         for(int i=0;i<net->layers[k].from_idx.size();i++){
//                             cat_input.push_back(net->layers[k + net->layers[k].from_idx[i]].output);
//                         }
//                         out = torch::cat(cat_input, 1);
//                     }
//                     else if(net->layers[k].name == "avg_pool2d"){
//                         out = F::avg_pool2d(out,F::AvgPool2dFuncOptions(3).stride(1).padding(1));
//                     }
//                     else if(net->layers[k].name == "max_pool2d"){
//                         out = F::max_pool2d(out,F::MaxPool2dFuncOptions(3).stride(2));
//                     }
//                     else{
//                         out = net->layers[k].layer.forward(inputs).toTensor();
//                     }
//                     net->layers[k].output = out;
//                 }
            
//                 k--;
//                 int record = net->layers[k].event_idx;
//                 // Stream에서 event 발생시 위치 표시
//                 // cudaEventRecord(nl->net->record[record], 0);
//                 cudaEventRecord(net->record[record], streams[net->H_L][stream_id[(net->layers[k].event_idx)%4]]);
//             }
//         }
//         else if(net->layers[k].name == "concat"){  //brach out
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
//     if(net->layers[k].event_idx >= 0){
// 		cudaEventSynchronize(net->record[net->layers[k].event_idx]);
// 		net->layers[k].output = out;
// 		net->layers[k].exe_success = true;
// 	}
//     net->layers[k].output = out;

// 	net->index = k;
// 	cond_i[net->index_n]=0;
// 	pthread_cond_signal(&cond_t[net->index_n]);

    
// 	pthread_mutex_unlock(&mutex_t[net->index_n]);
// 	return out;		
// }


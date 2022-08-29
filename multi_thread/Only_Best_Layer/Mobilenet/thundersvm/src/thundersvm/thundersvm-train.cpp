// //
// // Created by jiashuai on 17-9-14.
// //


// // #include <thundersvm/util/log.h>
// // #include <thundersvm/model/svc.h>
// // #include <thundersvm/model/svr.h>
// // #include <thundersvm/model/oneclass_svc.h>
// // #include <thundersvm/model/nusvc.h>
// // #include <thundersvm/model/nusvr.h>
// // #include <thundersvm/util/metric.h>
// // #include "thundersvm/cmdparser.h"

// //////////test.cpp header files//////////
// #include <torch/script.h>
// #include <torch/torch.h>
// #include <typeinfo>
// #include <iostream>
// #include <inttypes.h>
// #include <functional>
// #include <memory>
// #include <stdlib.h>
// #include <pthread.h>
// #include <cuda_runtime.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>
// #include <c10/cuda/CUDAStream.h>
// #include <c10/cuda/CUDAFunctions.h>
// #include <time.h>

// #include "thundersvm/test.h"
// /////test.cpp header files//////////

// /////from test.cpp//////////
// #define n_dense 12
// #define n_res 12
// #define n_alex 12
// #define n_vgg 12
// #define n_wide 12
// #define n_squeeze 0    // MAX 32
// #define n_mobile 0
// #define n_mnasnet 0
// #define n_inception 0 // MAX 16
// #define n_shuffle 0    // MAX 32
// #define n_resX 10

// #define n_threads 4 // inception의 병렬화 실행을 위한 최소한의 thread 갯수
// #define WARMING 4

// extern void *predict_alexnet(Net *input);
// extern void *predict_vgg(Net *input);
// extern void *predict_resnet(Net *input);
// extern void *predict_densenet(Net *input);
// extern void *predict_squeeze(Net *input);
// extern void *predict_mobilenet(Net *input);
// extern void *predict_MNASNet(Net *input);
// extern void *predict_inception(Net *input);
// extern void *predict_shuffle(Net *input);

// namespace F = torch::nn::functional;
// //using namespace std;
// using std::cout;
// using std::endl;
// using std::cerr;

// void print_script_module(const torch::jit::script::Module& module, size_t spaces) {
//     for (const auto& sub_module : module.named_children()) {
//         if (!sub_module.name.empty()) {
//             std::cout << std::string(spaces, ' ') << sub_module.value.type()->name().value().name()
//                 << " " << sub_module.name << "\n";    
//         }
//         print_script_module(sub_module.value, spaces + 2);
//     }
// }

// void print_vector(vector<int> v){
// 	for(int i=0;i<v.size();i++){
// 		cout<<v[i]<<" ";
// 	}
// 	cout<<"\n";
// }

// threadpool thpool;
// pthread_cond_t* cond_t;
// pthread_mutex_t* mutex_t;
// int* cond_i;

// std::vector<std::vector <at::cuda::CUDAStream>> streams;
// int16_t GPU_NUM = 1;
// /////from test.cpp//////////


// #ifdef _WIN32
// INITIALIZE_EASYLOGGINGPP
// #endif

// int main(int argc, char **argv) {
//     //////////from test.cpp/////////
//       c10::cuda::set_device(GPU_NUM);
//   torch::Device device = {at::kCUDA,GPU_NUM};
//   int n_all = n_alex + n_vgg + n_res + n_dense + n_wide + n_squeeze + n_mobile + n_mnasnet + n_inception + n_shuffle + n_resX;
//   static int stream_index_L = 0;
//   static int stream_index_H = 0;
//   static int branch_index_L = 31;
//   static int branch_index_H = 31;
//   static int net_priority_L = n_all/2; // LOW index model get HIHG priority
//   static int net_priority_H = n_all;

//   float time;
//   cudaEvent_t start, end;

//   thpool = thpool_init(n_threads, n_all);

//   streams.resize(2); // streams[][] 형식으로 사용할 것


//   /* stream 생성 */

//   for(int i=0; i<n_streamPerPool; i++){
//     streams[1].push_back(at::cuda::getStreamFromPool(true,0)); //high priority stream  (priority = -1)
//   }
//   for(int i=0; i<n_streamPerPool; i++){
//     streams[0].push_back(at::cuda::getStreamFromPool(false,0)); //low priority stream  (priority = 0)
//   }



//   torch::jit::script::Module denseModule;
//   torch::jit::script::Module resModule;
//   torch::jit::script::Module alexModule;
//   torch::jit::script::Module vggModule;
//   torch::jit::script::Module wideModule;
//   torch::jit::script::Module squeezeModule;
//   torch::jit::script::Module mobileModule;
//   torch::jit::script::Module mnasModule;
//   torch::jit::script::Module inceptionModule;
//   torch::jit::script::Module shuffleModule;
//   torch::jit::script::Module resXModule;

//   /* Model Load */
//   try {
//     	denseModule = torch::jit::load("/home/kmsjames/very-big-storage/joo/densenet_model.pt");
//       denseModule.to(device);

//     	resModule = torch::jit::load("/home/kmsjames/very-big-storage/joo/resnet_model.pt");
//       resModule.to(device);

//     	alexModule = torch::jit::load("/home/kmsjames/very-big-storage/joo/alexnet_model.pt");
//       alexModule.to(device);
  
//     	vggModule = torch::jit::load("/home/kmsjames/very-big-storage/joo/vgg_model.pt");
//       vggModule.to(device);

//     	wideModule = torch::jit::load("/home/kmsjames/very-big-storage/joo/wideresnet_model.pt");
//       wideModule.to(device);
 
//     	squeezeModule = torch::jit::load("/home/kmsjames/very-big-storage/joo/squeeze_model.pt");
//       squeezeModule.to(device);

//     	mobileModule = torch::jit::load("/home/kmsjames/very-big-storage/joo/mobilenet_model.pt");
//       mobileModule.to(device);

//     	mnasModule = torch::jit::load("/home/kmsjames/very-big-storage/joo/mnasnet_model.pt");
//       mnasModule.to(device);

//     	inceptionModule = torch::jit::load("/home/kmsjames/very-big-storage/joo/inception_model.pt");
//       inceptionModule.to(device);

//     	shuffleModule = torch::jit::load("/home/kmsjames/very-big-storage/joo/shuffle_model.pt");
//       shuffleModule.to(device);

//     	resXModule = torch::jit::load("/home/kmsjames/very-big-storage/joo/resnext_model.pt");
//       resXModule.to(device);
//   }
//   catch (const c10::Error& e) {
//     cerr << "error loading the model\n";
//     return -1;
//   }
//   cout<<"***** Model Load compelete *****"<<"\n";

//   cond_t = (pthread_cond_t *)malloc(sizeof(pthread_cond_t) * n_all);
//   mutex_t = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) * n_all);
//   cond_i = (int *)malloc(sizeof(int) * n_all);

//   for (int i = 0; i < n_all; i++)
//   {
//       pthread_cond_init(&cond_t[i], NULL);
//       pthread_mutex_init(&mutex_t[i], NULL);
//       cond_i[i] = 0;
//   }


//   vector<torch::jit::IValue> inputs;
//   vector<torch::jit::IValue> inputs2;
//   //module.to(at::kCPU);
   
//   torch::Tensor x = torch::ones({1, 3, 224, 224}).to(at::kCUDA);
//   torch::Tensor x2 = torch::ones({1, 3, 299, 299}).to(at::kCUDA);
//   inputs.push_back(x);
//   inputs2.push_back(x2);
  
  
//   Net net_input_dense[n_dense];
//   Net net_input_res[n_res];
//   Net net_input_alex[n_alex];
//   Net net_input_vgg[n_vgg];
//   Net net_input_wide[n_wide];
//   Net net_input_squeeze[n_squeeze];
//   Net net_input_mobile[n_mobile];
//   Net net_input_mnasnet[n_mnasnet];
//   Net net_input_inception[n_inception];
//   Net net_input_shuffle[n_shuffle];
//   Net net_input_resX[n_resX];

//   pthread_t networkArray_dense[n_dense];
//   pthread_t networkArray_res[n_res];
//   pthread_t networkArray_alex[n_alex];
//   pthread_t networkArray_vgg[n_vgg];
//   pthread_t networkArray_wide[n_wide];
//   pthread_t networkArray_squeeze[n_squeeze];
//   pthread_t networkArray_mobile[n_mobile];
//   pthread_t networkArray_mnasnet[n_mnasnet];
//   pthread_t networkArray_inception[n_inception];
//   pthread_t networkArray_shuffle[n_shuffle];
//   pthread_t networkArray_resX[n_resX];

//   for(int i=0;i<n_dense;i++){
// 	  get_submodule_densenet(denseModule, net_input_dense[i]);
//     std::cout << "End get submodule_densenet "<< i << "\n";
//     net_input_dense[i].input = inputs;
//     net_input_dense[i].name = "DenseNet";
//     net_input_dense[i].flatten = net_input_dense[i].layers.size()-1;
//     net_input_dense[i].index_n = i;
//     // priQ는 높은 priority값이 높은 우선순위를 가짐
//     if(i < n_dense/2){  // HIGH priority stream
//       net_input_dense[i].H_L = 1; 
//       net_input_dense[i].index_s = stream_index_H;
//       net_input_dense[i].priority = net_priority_H;
//       stream_index_H+=1;
//       net_priority_H-=1;
//     }
//     else{               // LOW prioirty stream
//       net_input_dense[i].H_L = 0; 
//       net_input_dense[i].index_s = stream_index_L;
//       net_input_dense[i].priority = net_priority_L;
//       stream_index_L+=1;
//       net_priority_L-=1;
//     }
//     for(int j=0;j<WARMING;j++){
//       predict_densenet(&net_input_dense[i]);
//       net_input_dense[i].input = inputs;
//     }
//     std::cout << "====== END DenseNet WARMUP ======" << std::endl;
//   }

//   for(int i=0;i<n_res;i++){
// 	  get_submodule_resnet(resModule, net_input_res[i]);
//     std::cout << "End get submodule_resnet "<< i << "\n";
//     net_input_res[i].name = "ResNet";
//     net_input_res[i].flatten = net_input_res[i].layers.size()-1;
// 	  net_input_res[i].input = inputs;
//     net_input_res[i].index_n = i+n_dense;
//     if(i < n_res/2){  // HIGH priority stream
//       net_input_res[i].H_L = 1; 
//       net_input_res[i].index_s = stream_index_H;
//       net_input_res[i].priority = net_priority_H;
//       stream_index_H+=1;
//       net_priority_H-=1;
//     }
//     else{               // LOW prioirty stream
//       net_input_res[i].H_L = 0; 
//       net_input_res[i].index_s = stream_index_L;
//       net_input_res[i].priority = net_priority_L;
//       stream_index_L+=1;
//       net_priority_L-=1;
//     }
//     for(int j=0;j<WARMING;j++){
//       predict_resnet(&net_input_res[i]);
//       net_input_res[i].input = inputs;
//     }
//     std::cout << "====== END ResNet WARMUP ======" << std::endl;
//   }

//   for(int i=0;i<n_alex;i++){
// 	  get_submodule_alexnet(alexModule, net_input_alex[i]);
//     std::cout << "End get submodule_alexnet " << i <<"\n";
// 	  net_input_alex[i].input = inputs;
//     net_input_alex[i].name = "AlexNet";
//     net_input_alex[i].flatten = net_input_alex[i].layers.size()-7;
//     net_input_alex[i].index_n = i+ n_res + n_dense;
//     if(i < n_alex/2){  // HIGH priority stream
//       net_input_alex[i].H_L = 1; 
//       net_input_alex[i].index_s = stream_index_H;
//       net_input_alex[i].priority = net_priority_H;
//       stream_index_H+=1;
//       net_priority_H-=1;
//     }
//     else{               // LOW prioirty stream
//       net_input_alex[i].H_L = 0; 
//       net_input_alex[i].index_s = stream_index_L;
//       net_input_alex[i].priority = net_priority_L;
//       stream_index_L+=1;
//       net_priority_L-=1;
//     }
//     for(int j=0;j<WARMING;j++){
//       predict_alexnet(&net_input_alex[i]);
//       net_input_alex[i].input = inputs;
//     }
//     std::cout << "====== END Alex WARMUP ======" << std::endl;
//   }

//   for(int i=0;i<n_vgg;i++){
// 	  get_submodule_vgg(vggModule, net_input_vgg[i]);
//     std::cout << "End get submodule_vgg " << i << "\n";
// 	  net_input_vgg[i].input = inputs;
//     net_input_vgg[i].name = "VGG";
//     net_input_vgg[i].flatten = net_input_vgg[i].layers.size()-7;
//     net_input_vgg[i].index_n = i + n_alex + n_res + n_dense;
//     if(i < n_vgg/2){  // HIGH priority stream
//       net_input_vgg[i].H_L = 1; 
//       net_input_vgg[i].index_s = stream_index_H;
//       net_input_vgg[i].priority = net_priority_H;
//       stream_index_H+=1;
//       net_priority_H-=1;
//     }
//     else{               // LOW prioirty stream
//       net_input_vgg[i].H_L = 0; 
//       net_input_vgg[i].index_s = stream_index_L;
//       net_input_vgg[i].priority = net_priority_L;
//       stream_index_L+=1;
//       net_priority_L-=1;
//     }
//     for(int j=0;j<WARMING;j++){
//       predict_vgg(&net_input_vgg[i]);
//       net_input_vgg[i].input = inputs;
//     }
//     std::cout << "====== END VGG WARMUP ======" << std::endl;
//   }

//   for(int i=0;i<n_wide;i++){
// 	  get_submodule_resnet(wideModule, net_input_wide[i]);
//     std::cout << "End get submodule_widenet "<< i << "\n";
// 	  net_input_wide[i].input = inputs;
//     net_input_wide[i].name = "WideResNet";
//     net_input_wide[i].flatten = net_input_wide[i].layers.size()-1;
//     net_input_wide[i].index_n = i+n_alex + n_res + n_dense + n_vgg;
//     if(i < n_wide/2){  // HIGH priority stream
//       net_input_wide[i].H_L = 1; 
//       net_input_wide[i].index_s = stream_index_H;
//       net_input_wide[i].priority = net_priority_H;
//       stream_index_H+=1;
//       net_priority_H-=1;
//     }
//     else{               // LOW prioirty stream
//       net_input_wide[i].H_L = 0; 
//       net_input_wide[i].index_s = stream_index_L;
//       net_input_wide[i].priority = net_priority_L;
//       stream_index_L+=1;
//       net_priority_L-=1;
//     }
//     for(int j=0;j<WARMING;j++){
//       predict_resnet(&net_input_wide[i]);
//       net_input_wide[i].input = inputs;
//     }
//     std::cout << "====== END wideRes WARMUP ======" << std::endl;
//   }

//   for(int i=0;i<n_squeeze;i++){
// 	  get_submodule_squeeze(squeezeModule, net_input_squeeze[i]);
//     std::cout << "End get submodule_squeezenet "<< i << "\n";
//     for(int j=0;j<2;j++){
//       cudaEvent_t event_temp;
//       cudaEventCreate(&event_temp);
//       net_input_squeeze[i].record.push_back(event_temp);
//     }
//     net_input_squeeze[i].name = "SqueezeNet";
//     net_input_squeeze[i].flatten = net_input_squeeze[i].layers.size()-1;
//     net_input_squeeze[i].n_all = n_all;
// 	  net_input_squeeze[i].input = inputs;
//     net_input_squeeze[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide;
//     if(i < n_squeeze/2){  // HIGH priority stream
//       net_input_squeeze[i].H_L = 1; 
//       net_input_squeeze[i].index_s = stream_index_H;
//       net_input_squeeze[i].index_b = branch_index_H;
//       net_input_squeeze[i].priority = net_priority_H;
//       stream_index_H+=1;
//       branch_index_H-=1;
//       net_priority_H-=1;
//     }
//     else{                 // LOW priority stream
//       net_input_squeeze[i].H_L = 0; 
//       net_input_squeeze[i].index_s = stream_index_L;
//       net_input_squeeze[i].index_b = branch_index_L;
//       net_input_squeeze[i].priority = net_priority_L;
//       stream_index_L+=1;
//       branch_index_L-=1;
//       net_priority_L-=1;
//     }
//     for(int j=0;j<WARMING;j++){
//       predict_squeeze(&net_input_squeeze[i]);
//       net_input_squeeze[i].input = inputs;
//       for(int n=0;n<net_input_squeeze[i].layers.size();n++){
//         net_input_squeeze[i].layers[n].exe_success = false;
//       }
//     }
//     std::cout << "====== END Squeeze WARMUP ======" << std::endl;
//   }

//   for(int i=0;i<n_mobile;i++){
// 	  get_submodule_mobilenet(mobileModule, net_input_mobile[i]);
//     std::cout << "End get submodule_mobilenet "<< i << "\n";
// 	  net_input_mobile[i].input = inputs;
//     net_input_mobile[i].name = "Mobile";
//     net_input_mobile[i].flatten = net_input_mobile[i].layers.size()-2;
//     net_input_mobile[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide + n_squeeze;
//     if(i < n_mobile/2){  // HIGH priority stream
//       net_input_mobile[i].H_L = 1; 
//       net_input_mobile[i].index_s = stream_index_H;
//       net_input_mobile[i].priority = net_priority_H;
//       stream_index_H+=1;
//       net_priority_H-=1;
//     }
//     else{               // LOW prioirty stream
//       net_input_mobile[i].H_L = 0; 
//       net_input_mobile[i].index_s = stream_index_L;
//       net_input_mobile[i].priority = net_priority_L;
//       stream_index_L+=1;
//       net_priority_L-=1;
//     }
//     for(int j=0;j<WARMING;j++){
//       predict_mobilenet(&net_input_mobile[i]);
//       net_input_mobile[i].input = inputs;
//     }
//     std::cout << "====== END Mobile WARMUP ======" << std::endl;
//   }

//   for(int i=0;i<n_mnasnet;i++){
// 	  get_submodule_MNASNet(mnasModule, net_input_mnasnet[i]);
//     std::cout << "End get submodule_mnasnet "<< i << "\n";
// 	  net_input_mnasnet[i].input = inputs;
//     net_input_mnasnet[i].name = "MNASNet";
//     net_input_mnasnet[i].flatten = net_input_mnasnet[i].layers.size()-2;
//     net_input_mnasnet[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide + n_squeeze + n_mobile;
//     if(i < n_mnasnet/2){  // HIGH priority stream
//       net_input_mnasnet[i].H_L = 1; 
//       net_input_mnasnet[i].index_s = stream_index_H;
//       net_input_mnasnet[i].priority = net_priority_H;
//       stream_index_H+=1;
//       net_priority_H-=1;
//     }
//     else{               // LOW prioirty stream
//       net_input_mnasnet[i].H_L = 0; 
//       net_input_mnasnet[i].index_s = stream_index_L;
//       net_input_mnasnet[i].priority = net_priority_L;
//       stream_index_L+=1;
//       net_priority_L-=1;
//     }
//     for(int j=0;j<WARMING;j++){
//       predict_MNASNet(&net_input_mnasnet[i]);
//       net_input_mnasnet[i].input = inputs;
//     }
//     std::cout << "====== END MNASnet WARMUP ======" << std::endl;
//   }

//   for(int i=0;i<n_inception;i++){
// 	  get_submodule_inception(inceptionModule, net_input_inception[i]);
//     std::cout << "End get submodule_inception "<< i << "\n";
//     for(int j=0;j<4;j++){
//       cudaEvent_t event_temp;
//       cudaEventCreate(&event_temp);
//       net_input_inception[i].record.push_back(event_temp);
//     }
//     net_input_inception[i].n_all = n_all;
// 	  net_input_inception[i].input = inputs2;
//     net_input_inception[i].name = "Inception_v3";
//     net_input_inception[i].flatten = net_input_inception[i].layers.size()-2;
//     net_input_inception[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide + n_squeeze + n_mobile + n_mnasnet;
//     if(i < n_inception/2){  // HIGH priority stream
//       net_input_inception[i].H_L = 1; 
//       net_input_inception[i].index_s = stream_index_H;
//       net_input_inception[i].index_b = branch_index_H;
//       net_input_inception[i].priority = net_priority_H;
//       stream_index_H+=1;
//       branch_index_H-=3;  
//       net_priority_H-=1;
//     }
//     else{                 // LOW priority stream
//       net_input_inception[i].H_L = 0; 
//       net_input_inception[i].index_s = stream_index_L;
//       net_input_inception[i].index_b = branch_index_L;
//       net_input_inception[i].priority = net_priority_L;
//       stream_index_L+=1;
//       branch_index_L-=3;
//       net_priority_L-=1;
//     }
    
//     for(int j=0;j<WARMING;j++){
//       predict_inception(&net_input_inception[i]);
//       net_input_inception[i].input = inputs2;
//       for(int n=0;n<net_input_inception[i].layers.size();n++){
//         net_input_inception[i].layers[n].exe_success = false;
//       }
//     }
//     std::cout << "====== END Inception v3 WARMUP ======" << std::endl;
//   }

//   for(int i=0;i<n_shuffle;i++){
// 	  get_submodule_shuffle(shuffleModule, net_input_shuffle[i]);
//     std::cout << "End get submodule_shuffle "<< i << "\n";
//     for(int j=0;j<2;j++){
//       cudaEvent_t event_temp;
//       cudaEventCreate(&event_temp);
//       net_input_shuffle[i].record.push_back(event_temp);
//     }
//     net_input_shuffle[i].n_all = n_all;
// 	  net_input_shuffle[i].input = inputs;
//     net_input_shuffle[i].name = "ShuffleNet";
//     net_input_shuffle[i].flatten = net_input_shuffle[i].layers.size()-1;
//     net_input_shuffle[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide + n_squeeze + n_mobile + n_mnasnet + n_inception;
//     if(i < n_shuffle/2){  // HIGH priority stream
//       net_input_shuffle[i].H_L = 1; 
//       net_input_shuffle[i].index_s = stream_index_H;
//       net_input_shuffle[i].index_b = branch_index_H;
//       net_input_shuffle[i].priority = net_priority_H;
//       stream_index_H+=1;
//       branch_index_H-=1;
//       net_priority_H-=1;
//     }
//     else{                 // LOW priority stream
//       net_input_shuffle[i].H_L = 0; 
//       net_input_shuffle[i].index_s = stream_index_L;
//       net_input_shuffle[i].index_b = branch_index_L;
//       net_input_shuffle[i].priority = net_priority_L;
//       stream_index_L+=1;
//       branch_index_L-=1;
//       net_priority_L-=1;
//     }
//     for(int j=0;j<WARMING;j++){
//       predict_shuffle(&net_input_shuffle[i]);
//       net_input_shuffle[i].input = inputs;
//       for(int n=0;n<net_input_shuffle[i].layers.size();n++){
//         net_input_shuffle[i].layers[n].exe_success = false;
//       }
//     }
//     std::cout << "====== END ShuffleNet WARMUP ======" << std::endl;
//   }

//   for(int i=0;i<n_resX;i++){
// 	  get_submodule_resnet(resXModule, net_input_resX[i]);
//     std::cout << "End get submodule_resnext "<< i << "\n";
// 	  net_input_resX[i].input = inputs;
//     net_input_resX[i].name = "ResNext";
//     net_input_resX[i].flatten = net_input_resX[i].layers.size()-1;
//     net_input_resX[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide + n_squeeze + n_mobile + n_mnasnet + n_inception + n_shuffle;
//     if(i < n_resX/2){  // HIGH priority stream
//       net_input_resX[i].H_L = 1; 
//       net_input_resX[i].index_s = stream_index_H;
//       net_input_resX[i].priority = net_priority_H;
//       stream_index_H+=1;
//       net_priority_H-=1;
//     }
//     else{               // LOW prioirty stream
//       net_input_resX[i].H_L = 0; 
//       net_input_resX[i].index_s = stream_index_L;
//       net_input_resX[i].priority = net_priority_L;
//       stream_index_L+=1;
//       net_priority_L-=1;
//     }
//     for(int j=0;j<WARMING;j++){
//       predict_resnet(&net_input_resX[i]);
//       net_input_resX[i].input = inputs;
//     }
//     std::cout << "====== END ResNext WARMUP ======" << std::endl;
//   }

//   /* time check */
//   cudaEventCreate(&start);
//   cudaEventCreate(&end);
//   cudaEventRecord(start);
//   for(int i=0;i<n_dense;i++){
//     if (pthread_create(&networkArray_dense[i], NULL, (void *(*)(void*))predict_densenet, &net_input_dense[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }

//   for(int i=0;i<n_res;i++){
//     if (pthread_create(&networkArray_res[i], NULL, (void *(*)(void*))predict_resnet, &net_input_res[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }

//   for(int i=0;i<n_alex;i++){
//     if (pthread_create(&networkArray_alex[i], NULL, (void *(*)(void*))predict_alexnet, &net_input_alex[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }

//   for(int i=0;i<n_vgg;i++){
// 	  if (pthread_create(&networkArray_vgg[i], NULL, (void *(*)(void*))predict_vgg, &net_input_vgg[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }

//   for(int i=0;i<n_wide;i++){
//     if (pthread_create(&networkArray_wide[i], NULL, (void *(*)(void*))predict_resnet, &net_input_wide[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }

//   for(int i=0;i<n_squeeze;i++){
//     if (pthread_create(&networkArray_squeeze[i], NULL, (void *(*)(void*))predict_squeeze, &net_input_squeeze[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }

//   for(int i=0;i<n_mobile;i++){
//     if (pthread_create(&networkArray_mobile[i], NULL, (void *(*)(void*))predict_mobilenet, &net_input_mobile[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }

//   for(int i=0;i<n_mnasnet;i++){
//     if (pthread_create(&networkArray_mnasnet[i], NULL, (void *(*)(void*))predict_MNASNet, &net_input_mnasnet[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }

//   for(int i=0;i<n_inception;i++){
//     if (pthread_create(&networkArray_inception[i], NULL, (void *(*)(void*))predict_inception, &net_input_inception[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }

//   for(int i=0;i<n_shuffle;i++){
//     if (pthread_create(&networkArray_shuffle[i], NULL, (void *(*)(void*))predict_shuffle, &net_input_shuffle[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }
//   for(int i=0;i<n_resX;i++){
//     if (pthread_create(&networkArray_resX[i], NULL, (void *(*)(void*))predict_resnet, &net_input_resX[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }

//   for (int i = 0; i < n_dense; i++){
//     pthread_join(networkArray_dense[i], NULL); // pthread_join : thread 종료를 기다리고 thread 종료 이후 다음 진행
//   }                                            // join된 thread(종료된 thread)는 모든 resource를 반납

                                            
//   for (int i = 0; i < n_res; i++){
//     pthread_join(networkArray_res[i], NULL);
//   }

//   for (int i = 0; i < n_alex; i++){
//     pthread_join(networkArray_alex[i], NULL);
//   }

//   for (int i = 0; i < n_vgg; i++){
//     pthread_join(networkArray_vgg[i], NULL);
//   }

//   for (int i = 0; i < n_wide; i++){
//     pthread_join(networkArray_wide[i], NULL);
//   }

//   for (int i = 0; i < n_squeeze; i++){
//     pthread_join(networkArray_squeeze[i], NULL);
//   }

//   for (int i = 0; i < n_mobile; i++){
//     pthread_join(networkArray_mobile[i], NULL);
//   }

//   for (int i = 0; i < n_mnasnet; i++){
//     pthread_join(networkArray_mnasnet[i], NULL);
//   }

//   for (int i = 0; i < n_inception; i++){
//     pthread_join(networkArray_inception[i], NULL);
//   }

//   for (int i = 0; i < n_shuffle; i++){
//     pthread_join(networkArray_shuffle[i], NULL);
//   }

//   for (int i = 0; i < n_resX; i++){
//     pthread_join(networkArray_resX[i], NULL);
//   }
//   cudaEventRecord(end);
//   cudaEventSynchronize(end);
//   cudaEventElapsedTime(&time, start, end);
//   std::cout << "Total EXE TIME = "<< time/1000<<"'s"<< std::endl;
//   /* resX time check end */

//   cudaDeviceSynchronize();
//   free(cond_t);
//   free(mutex_t);
//   free(cond_i);
//   //////////from test.cpp//////////

//     try {
// 		el::Loggers::addFlag(el::LoggingFlag::FixedTimeFormat);
//         CMDParser parser;
//         parser.parse_command_line(argc, argv);
//         DataSet train_dataset;
//         train_dataset.load_from_file(parser.svmtrain_input_file_name);
//         std::shared_ptr<SvmModel> model;
//         switch (parser.param_cmd.svm_type) {
//             case SvmParam::C_SVC:
//                 model.reset(new SVC());
//                 LOG(INFO) << "training C-SVC";
//                 LOG(INFO) << "C = " << parser.param_cmd.C;
//                 break;
//             case SvmParam::NU_SVC:
//                 model.reset(new NuSVC());
//                 LOG(INFO) << "training nu-SVC";
//                 LOG(INFO) << "nu = " << parser.param_cmd.nu;
//                 break;
//             case SvmParam::ONE_CLASS:
//                 model.reset(new OneClassSVC());
//                 LOG(INFO) << "training one-class SVM";
//                 LOG(INFO) << "C = " << parser.param_cmd.C;
//                 break;
//             case SvmParam::EPSILON_SVR:
//                 model.reset(new SVR());
//                 LOG(INFO) << "training epsilon-SVR";
//                 LOG(INFO) << "C = " << parser.param_cmd.C << " p = " << parser.param_cmd.p;
//                 break;
//             case SvmParam::NU_SVR:
//                 model.reset(new NuSVR());
//                 LOG(INFO) << "training nu-SVR";
//                 LOG(INFO) << "nu = " << parser.param_cmd.nu;
//                 break;
//         }

//         //todo add this to check_parameter method
//         if (parser.param_cmd.svm_type == SvmParam::NU_SVC) {
//             train_dataset.group_classes();
//             for (int i = 0; i < train_dataset.n_classes(); ++i) {
//                 int n1 = train_dataset.count()[i];
//                 for (int j = i + 1; j < train_dataset.n_classes(); ++j) {
//                     int n2 = train_dataset.count()[j];
//                     if (parser.param_cmd.nu * (n1 + n2) / 2 > min(n1, n2)) {
//                         printf("specified nu is infeasible\n");
//                         return 1;
//                     }
//                 }
//             }
//         }
//         if (parser.param_cmd.kernel_type != SvmParam::LINEAR)
//             if (!parser.gamma_set) {
//                 parser.param_cmd.gamma = 1.f / train_dataset.n_features();
//                 LOG(WARNING) << "using default gamma=" << parser.param_cmd.gamma;
//             } else {
//                 LOG(INFO) << "gamma = " << parser.param_cmd.gamma;
//             }

// #ifdef USE_CUDA
//         CUDA_CHECK(cudaSetDevice(parser.gpu_id));
// #endif

//         vector<float_type> predict_y;
//         if (parser.do_cross_validation) {
//             predict_y = model->cross_validation(train_dataset, parser.param_cmd, parser.nr_fold);
//         } else {
//             model->train(train_dataset, parser.param_cmd);
//             LOG(INFO) << "training finished";
//             model->save_to_file(parser.model_file_name);
//          //   LOG(INFO) << "evaluating training score";
//          //   predict_y = model->predict(train_dataset.instances(), -1);
//         }

//         //perform svm testing
//         if(parser.do_cross_validation) {
//             std::shared_ptr<Metric> metric;
//             switch (parser.param_cmd.svm_type) {
//                 case SvmParam::C_SVC:
//                 case SvmParam::NU_SVC: {
//                     metric.reset(new Accuracy());
//                     break;
//                 }
//                 case SvmParam::EPSILON_SVR:
//                 case SvmParam::NU_SVR: {
//                     metric.reset(new MSE());
//                     break;
//                 }
//                 case SvmParam::ONE_CLASS: {
//                 }
//             }
//             if (metric) {
//                 std::cout << "Cross " << metric->name() << " = " << metric->score(predict_y, train_dataset.y()) << std::endl;
//             }
//         }

//     }
//     catch (std::bad_alloc &) {
//         LOG(FATAL) << "out of memory, you may try \"-m memory size\" to constrain memory usage";
//         exit(EXIT_FAILURE);
//     }
//     catch (std::exception const &x) {
//         LOG(FATAL) << x.what();
//         exit(EXIT_FAILURE);
//     }
//     catch (...) {
//         LOG(FATAL) << "unknown error";
//         exit(EXIT_FAILURE);
//     }
// }


#ifndef MOBILE_H
#define MOBILE_H

#include "net.h"
#include "test.h"
#include "thpool.h"


void get_submodule_mobilenet(torch::jit::script::Module module, Net &net);
void *predict_mobilenet(Net *input);
at::Tensor forward_mobilenet(Net *net, int idx);

void *vi_job(Visvm * vi_svm);
#endif


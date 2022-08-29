#ifndef ALEX_H
#define ALEX_H

#include "net.h"
#include "test.h"
#include "thpool.h"


void get_submodule_alexnet(torch::jit::script::Module module, Net &child);
void *predict_alexnet(Net *input);
at::Tensor forward_alexnet(Net *net, int idx);

void *vi_job(Visvm * vi_svm);
#endif
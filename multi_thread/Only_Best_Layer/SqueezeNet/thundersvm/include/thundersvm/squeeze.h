#ifndef SQUEEZE_H
#define SQUEEZE_H

#include "net.h"
#include "test.h"
#include "thpool.h"

void get_submodule_squeeze(torch::jit::script::Module module, Net &net);
void *predict_squeeze_warm(Net *input);
void *predict_squeeze(Net *input);
at::Tensor forward_squeeze(Net *net, int idx);
at::Tensor forward_squeeze_1(Net *net, int idx);
at::Tensor forward_squeeze_2(Net *net, int idx);

void *vi_job(Visvm * vi_svm);
#endif

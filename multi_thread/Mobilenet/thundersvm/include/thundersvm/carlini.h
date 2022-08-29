#ifndef CARLINI_H
#define CARLINI_H

#include "net.h"
#include "test.h"
#include "thpool.h"


void get_submodule_carlini(torch::jit::script::Module module, Net &child);
void *predict_carlini(Net *input);
torch::jit::IValue forward_carlini(Net * net, int idx);

void *vi_job(Visvm * vi_svm);


#endif


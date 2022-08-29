#ifndef INCEPTION_H
#define INCEPTION_H

#include "net.h"
#include "test.h"
#include "thpool.h"


void get_submodule_inception(torch::jit::script::Module module, Net &child);
void *predict_inception(Net *input);
at::Tensor forward_inception(Net *net, int idx);
#endif


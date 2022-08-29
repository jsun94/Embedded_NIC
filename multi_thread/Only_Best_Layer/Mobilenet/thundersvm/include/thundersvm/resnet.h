#ifndef RESNET18_H
#define RESNET18_H

#include "net.h"
#include "test.h"
#include "thpool.h"

void get_submodule_resnet(torch::jit::script::Module module, Net &net);
void *predict_resnet(Net *input);
at::Tensor forward_resnet(Net *net, int idx);

#endif

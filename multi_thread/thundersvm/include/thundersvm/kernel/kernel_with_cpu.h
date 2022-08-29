//
// skycs
//

#ifndef THUNDERSVM_KERNEL_WITH_CPU_H
#define THUNDERSVM_KERNEL_WITH_CPU_H

#include "thundersvm/thundersvm.h"
#include <thundersvm/clion_cuda.h>
#include <thundersvm/syncarray.h>

namespace svm_kernel {
    void sum_kernel_values_with_cpu(const SyncArray<float_type> &coef, int total_sv, const SyncArray<int> &sv_start,
                           const SyncArray<int> &sv_count, const SyncArray<float_type> &rho,
                           const SyncArray<kernel_type> &k_mat,
                           SyncArray<float_type> &dec_values, int n_classes, int n_instances, int n_core);
}
#endif //THUNDERSVM_KERNELMATRIX_KERNEL_H


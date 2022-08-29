//skycs
#include <thundersvm/kernel/kernel_with_cpu.h>
#include <omp.h>
namespace svm_kernel {
    void sum_kernel_values_with_cpu(const SyncArray<float_type> &coef, int total_sv, const SyncArray<int> &sv_start,
                            const SyncArray<int> &sv_count, const SyncArray<float_type> &rho,
                            const SyncArray<kernel_type> &k_mat,
                            SyncArray<float_type> &dec_values, int n_classes, int n_instances, int n_core){
            const int *sv_start_data = sv_start.host_data();
            const int *sv_count_data = sv_count.host_data();
            const float_type *coef_data = coef.host_data();
            const kernel_type *k_mat_data = k_mat.host_data();
            float_type *dec_values_data = dec_values.host_data();
            const float_type* rho_data = rho.host_data();
        #pragma omp parallel num_threads(n_core)
        {
    #pragma omp parallel for schedule(guided)
            for (int idx = 0; idx < n_instances; idx++) {
                int k = 0;
                int n_binary_models = n_classes * (n_classes - 1) / 2;
                for (int i = 0; i < n_classes; ++i) {
                    for (int j = i + 1; j < n_classes; ++j) {
                        int si = sv_start_data[i];
                        int sj = sv_start_data[j];
                        int ci = sv_count_data[i];
                        int cj = sv_count_data[j];
                        const float_type *coef1 = &coef_data[(j - 1) * total_sv];
                        const float_type *coef2 = &coef_data[i * total_sv];
                        const kernel_type *k_values = &k_mat_data[idx * total_sv];
                        double sum = 0;
    #pragma omp parallel for reduction(+:sum)
                        for (int l = 0; l < ci; ++l) {
                            sum += coef1[si + l] * k_values[si + l];
                        }
    #pragma omp parallel for reduction(+:sum)
                        for (int l = 0; l < cj; ++l) {
                            sum += coef2[sj + l] * k_values[sj + l];
                        }
                        dec_values_data[idx * n_binary_models + k] = sum - rho_data[k];
                        k++;
                    }
                }
            }
        }
    }
}
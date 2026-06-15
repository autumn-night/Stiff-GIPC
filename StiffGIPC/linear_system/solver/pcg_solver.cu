#include <linear_system/solver/pcg_solver.h>
#include <gipc/utils/timer.h>
#include <gipc/statistics.h>
#include <cuda_tools/cuda_tools.h>



__global__ void PCG_vdv_Reduction(double* squeue, const double* a, const double* b, int numbers)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= numbers)
        return;

    double temp = a[idx] * b[idx];

    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    //double nextTp;
    int    warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        temp = tep[threadIdx.x];
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}



__global__ void add_reduction(double* mem, int numbers)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;
    extern __shared__ double tep[];
    if(idx >= numbers)
        return;
    double temp = mem[idx];
    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    int    warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        temp = tep[threadIdx.x];
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        mem[blockIdx.x] = temp;
    }
}



__global__ void update_vector_dx_r(
    double* dx, double* r, const double* c, const double* q, double alpha, int numbers)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= numbers)
        return;
    dx[idx] = dx[idx] + alpha * c[idx];
    r[idx]  = r[idx] - alpha * q[idx];
}

__global__ void update_vector_c(
    double* c, const double* s, double beta, int numbers)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= numbers)
        return;
    c[idx] = s[idx] + beta * c[idx];
}


double My_PCG_General_v_v_Reduction_Algorithm(double* temp, double* A, double* B, int vertexNum)
{

    int numbers = vertexNum;
    if(numbers < 1)
        return 0;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);
    PCG_vdv_Reduction<<<blockNum, threadNum, sharedMsize>>>(temp, A, B, numbers);


    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        add_reduction<<<blockNum, threadNum, sharedMsize>>>(temp, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    double result;
    cudaMemcpy(&result, temp, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}

namespace gipc
{
PCGSolver::PCGSolver(const PCGSolverConfig& cfg)
    : m_config(cfg)
{
}
SizeT PCGSolver::solve(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b)
{
    Timer timer{"pcg"};

    // Reset sub-phase accumulators for this solve
    m_time_preconditioner_apply = 0.0;
    m_time_spmv                 = 0.0;
    m_time_dot                  = 0.0;
    m_time_axpby                = 0.0;

    x.buffer_view().fill(0);
    z.resize(b.size());
    p.resize(b.size());
    r.resize(b.size());
    //temp.resize(b.size());
    Ap.resize(b.size());
    auto iter = pcg(x, b, m_config.max_iter_ratio * b.size());

    // Write accumulated sub-phase times into the current Newton step's statistics
    auto& json = gipc::Statistics::instance().at_current_frame();
    if(json.contains("newton") && json["newton"].is_array() && !json["newton"].empty())
    {
        auto& pcg_json = json["newton"].back()["pcg"];
        pcg_json["preconditioner_apply_ms"] = m_time_preconditioner_apply;
        pcg_json["spmv_ms"]                 = m_time_spmv;
        pcg_json["dot_ms"]                  = m_time_dot;
        pcg_json["axpby_ms"]                = m_time_axpby;
    }

    return iter;
}


SizeT PCGSolver::pcg(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b, SizeT max_iter)
{
    // Create CUDA events (reused across all iterations)
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    SizeT k = 0;

    r.buffer_view().copy_from(b.buffer_view());

    Float alpha, beta, rz, rz0;

    {
        cudaEventRecord(ev_start, 0);
        apply_preconditioner(z, r);
        cudaEventRecord(ev_stop, 0);
        cudaEventSynchronize(ev_stop);
        float phase_ms = 0;
        cudaEventElapsedTime(&phase_ms, ev_start, ev_stop);
        m_time_preconditioner_apply += phase_ms;
    }

    {
        cudaEventRecord(ev_start, 0);
        rz = My_PCG_General_v_v_Reduction_Algorithm(p.buffer_view().data(),
                                                    r.buffer_view().data(),
                                                    z.buffer_view().data(),
                                                    z.size());
        cudaEventRecord(ev_stop, 0);
        cudaEventSynchronize(ev_stop);
        float phase_ms = 0;
        cudaEventElapsedTime(&phase_ms, ev_start, ev_stop);
        m_time_dot += phase_ms;
    }

    p   = z;
    rz0 = rz;

    for(k = 1; k < max_iter; ++k)
    {
        {
            cudaEventRecord(ev_start, 0);
            // Ap = A * p
            spmv(p.cview(), Ap.view());
            cudaEventRecord(ev_stop, 0);
            cudaEventSynchronize(ev_stop);
            float phase_ms = 0;
            cudaEventElapsedTime(&phase_ms, ev_start, ev_stop);
            m_time_spmv += phase_ms;
        }

        {
            cudaEventRecord(ev_start, 0);

            Float dot_res =
                My_PCG_General_v_v_Reduction_Algorithm(z.buffer_view().data(),
                                                       p.buffer_view().data(),
                                                       Ap.buffer_view().data(),
                                                       z.size());

            alpha = rz / dot_res;
            cudaEventRecord(ev_stop, 0);
            cudaEventSynchronize(ev_stop);
            float phase_ms = 0;
            cudaEventElapsedTime(&phase_ms, ev_start, ev_stop);
            m_time_dot += phase_ms;
        }

        {
            cudaEventRecord(ev_start, 0);
            LaunchCudaKernal_default(z.size(),
                                     256,
                                     0,
                                     update_vector_dx_r,
                                     x.buffer_view().data(),
                                     r.buffer_view().data(),
                                     (const double*)p.buffer_view().data(),
                                     (const double*)Ap.buffer_view().data(),
                                     alpha,
                                     (int)z.size());
            cudaEventRecord(ev_stop, 0);
            cudaEventSynchronize(ev_stop);
            float phase_ms = 0;
            cudaEventElapsedTime(&phase_ms, ev_start, ev_stop);
            m_time_axpby += phase_ms;
        }

        if(std::abs(rz) <= m_config.global_tol_rate * rz0)
            break;

        {
            cudaEventRecord(ev_start, 0);
            apply_preconditioner(z, r);
            cudaEventRecord(ev_stop, 0);
            cudaEventSynchronize(ev_stop);
            float phase_ms = 0;
            cudaEventElapsedTime(&phase_ms, ev_start, ev_stop);
            m_time_preconditioner_apply += phase_ms;
        }

        Float rz_new = 0;
        {
            cudaEventRecord(ev_start, 0);
            rz_new = My_PCG_General_v_v_Reduction_Algorithm(Ap.buffer_view().data(),
                                                            r.buffer_view().data(),
                                                            z.buffer_view().data(),
                                                            z.size());
            cudaEventRecord(ev_stop, 0);
            cudaEventSynchronize(ev_stop);
            float phase_ms = 0;
            cudaEventElapsedTime(&phase_ms, ev_start, ev_stop);
            m_time_dot += phase_ms;
        }

        beta = rz_new / rz;

        {
            cudaEventRecord(ev_start, 0);
            LaunchCudaKernal_default(z.size(),
                                     256,
                                     0,
                                     update_vector_c,
                                     p.buffer_view().data(),
                                     (const double*)z.buffer_view().data(),
                                     beta,
                                     (int)z.size());
            cudaEventRecord(ev_stop, 0);
            cudaEventSynchronize(ev_stop);
            float phase_ms = 0;
            cudaEventElapsedTime(&phase_ms, ev_start, ev_stop);
            m_time_axpby += phase_ms;
        }

        rz = rz_new;
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return k;
}

}  // namespace gipc

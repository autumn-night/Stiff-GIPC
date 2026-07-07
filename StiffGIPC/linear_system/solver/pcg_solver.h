#pragma once
#include <linear_system/linear_system/i_linear_system_solver.h>

namespace gipc
{
class PCGSolverConfig
{
  public:
    /**
     * \brief the maximum number of iterations will be:
     *  dof * max_iter_ratio
     */
    Float max_iter_ratio  = 0.3;
    Float global_tol_rate = 1e-4;
    bool  use_bsr         = true;
};

class PCGSolver : public IterativeSolver
{
    using DeviceDenseVector = muda::DeviceDenseVector<Float>;

  public:
    PCGSolver(const PCGSolverConfig& cfg);
    virtual ~PCGSolver() = default;

    void config(const PCGSolverConfig& config) { this->m_config = config; }
    const auto& config() const { return this->m_config; }

    // Step 3: allow dynamic tolerance adjustment for inexact Newton
    void set_tolerance(Float tol) { m_config.global_tol_rate = tol; }

  private:

    DeviceDenseVector z;   // preconditioned residual
    DeviceDenseVector r;   // residual
    DeviceDenseVector p;   // search direction
    DeviceDenseVector Ap;  // A*p
    PCGSolverConfig   m_config;

    // Lightweight accumulated timing (milliseconds) for PCG sub-phases,
    // collected without cudaDeviceSynchronize to avoid per-iteration sync overhead.
    double m_time_preconditioner_apply = 0.0;
    double m_time_spmv                 = 0.0;
    double m_time_dot                  = 0.0;
    double m_time_axpby                = 0.0;

  protected:
    SizeT solve(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b) override;

  private:
    SizeT pcg(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b, SizeT max_iter);
};
}  // namespace gipc

#pragma once

#include <string>

class GIPC;

namespace app::common
{
struct SimulationSettings
{
    double density                        = 1e3;
    double poisson_rate                   = 0.49;
    double friction_rate                  = 0.4;
    double ground_friction_rate           = 0.4;
    double cloth_thickness                = 1e-3;
    double cloth_young_modulus            = 1e6;
    double bend_young_modulus             = 1e5;
    double cloth_density                  = 2e2;
    double strain_rate                    = 100;
    double soft_motion_rate               = 1.0;
    double collision_detection_buff_scale = 1.0;
    double motion_rate                    = 1.0;
    double ipc_dt                         = 1e-2;
    double pcg_threshold                  = 1e-4;
    double newton_solver_threshold        = 1e-2;
    double relative_dhat                  = 1e-3;
};

SimulationSettings load_simulation_settings(const std::string& path);

void apply_simulation_settings(GIPC& ipc,
                               double& collision_detection_buff_scale,
                               double& motion_rate,
                               const SimulationSettings& settings);
}  // namespace app::common

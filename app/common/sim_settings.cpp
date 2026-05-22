#include "app/common/sim_settings.h"

#include <GIPC.cuh>

#include <fstream>
#include <iostream>

namespace app::common
{
SimulationSettings load_simulation_settings(const std::string& path)
{
    SimulationSettings settings;

    std::ifstream infile(path, std::ifstream::in);
    if(!infile.is_open())
    {
        std::cerr << "Warning: failed loading settings from `" << path
                  << "`, fallback to defaults." << std::endl;
        return settings;
    }

    char ignore_token[256];
    infile >> ignore_token >> settings.density;
    infile >> ignore_token >> settings.poisson_rate;
    infile >> ignore_token >> settings.friction_rate;
    infile >> ignore_token >> settings.ground_friction_rate;
    infile >> ignore_token >> settings.cloth_thickness;
    infile >> ignore_token >> settings.cloth_young_modulus;
    infile >> ignore_token >> settings.bend_young_modulus;
    infile >> ignore_token >> settings.cloth_density;
    infile >> ignore_token >> settings.strain_rate;
    infile >> ignore_token >> settings.soft_motion_rate;
    infile >> ignore_token >> settings.collision_detection_buff_scale;
    infile >> ignore_token >> settings.motion_rate;
    infile >> ignore_token >> settings.ipc_dt;
    infile >> ignore_token >> settings.pcg_threshold;
    infile >> ignore_token >> settings.newton_solver_threshold;
    infile >> ignore_token >> settings.relative_dhat;
    return settings;
}

void apply_simulation_settings(GIPC& ipc,
                               double& collision_detection_buff_scale,
                               double& motion_rate,
                               const SimulationSettings& settings)
{
    ipc.density                 = settings.density;
    ipc.PoissonRate             = settings.poisson_rate;
    ipc.frictionRate            = settings.friction_rate;
    ipc.gd_frictionRate         = settings.ground_friction_rate;
    ipc.clothThickness          = settings.cloth_thickness;
    ipc.clothYoungModulus       = settings.cloth_young_modulus;
    ipc.bendYoungModulus        = settings.bend_young_modulus;
    ipc.clothDensity            = settings.cloth_density;
    ipc.strainRate              = settings.strain_rate;
    ipc.softMotionRate          = settings.soft_motion_rate;
    ipc.IPC_dt                  = settings.ipc_dt;
    ipc.pcg_threshold           = settings.pcg_threshold;
    ipc.Newton_solver_threshold = settings.newton_solver_threshold;
    ipc.relative_dhat           = settings.relative_dhat;

    collision_detection_buff_scale = settings.collision_detection_buff_scale;
    motion_rate                    = settings.motion_rate;
}
}  // namespace app::common

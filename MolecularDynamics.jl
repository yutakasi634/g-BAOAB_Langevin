using Printf
using Random
using Distributions
using LinearAlgebra

# simulation meta information
const time_step = 0.8
const total_step = 5000
const dump_step = 1
const rattle_tolerance = 1.0e-6
const rattle_max_iter = 500

# physical constants
const kB = 0.0019872 # kcal/ mol K

# system paraneters
const lennard_jones_eps = 0.6
const lennard_jones_sigma = 2.0
const gamma = 0.01
const temperature = 300.0

# meso-scale system parameters
const patch_particle_num = 6
const particle_num = patch_particle_num * 2
const core_patch_dist = 4.0
const core_patch_bond_coef = 10.0
const inverse_power_coef = 0.2
const inverse_power_n = 5

# box parameters
const box_side_length        = 80.0
const box_eps                = 0.6
const box_sigma              = 4.0

# random number generator paramters
const rng = MersenneTwister(1234)

# pre calculation
const half_time_step = time_step / 2
const time_step2 = time_step * time_step
const one_minus_gammah_over2 = 1 - gamma * time_step * 0.5
const mass_vec     = zeros(1, particle_num)
mass_vec[1:2:particle_num] .= 100.0
mass_vec[2:2:particle_num] .= 20.0
const inv_mass_vec = 1 ./ mass_vec
const sqrt_inv_mass_vec = sqrt.(inv_mass_vec)
const noise_coef_vec = sqrt.(2gamma * kB * temperature / time_step .* inv_mass_vec)
const baoab_c_1 = ℯ^(-gamma * time_step)
const baoab_c_3 = sqrt(kB * temperature * (1 - baoab_c_1^2))
const core_patch_dist2 = core_patch_dist * core_patch_dist

log_file = open("logfile.log", "w")

include("force_energy_calculation.jl")
include("constraints.jl")
include("integrations.jl")

function main()
    println("main start")

    # meso-scale system initialization
    coord_vec = zeros(3, particle_num)
    side_num = ceil(patch_particle_num^(1/3))
    space = box_side_length / (side_num + 1)
    box_side_half = box_side_length / 2
    box_edge_coord = [-box_side_half, -box_side_half, -box_side_half]

    patch_particle_count = 0
    for x_idx in 1:side_num, y_idx in 1:side_num, z_idx in 1:side_num
        core_coord = box_edge_coord + [x_idx * space, y_idx * space, z_idx * space]
        coord_vec[:, patch_particle_count * 2 + 1] = core_coord
        coord_vec[:, patch_particle_count * 2 + 2] = core_coord + [core_patch_dist, 0, 0]
        patch_particle_count += 1
        if patch_particle_count == patch_particle_num
            break
        end
    end

    # pre preparation
    velocity_vec = zeros(3, particle_num)
    frame_vec_for_dump = []
    vel_vec_for_dump = []
    energy_vec_for_dump = []
    push!(frame_vec_for_dump, coord_vec)

    # system integration
    acceleration_vec = calculate_force(coord_vec) .* inv_mass_vec
    # loop start
    for step_idx in 1:total_step
        coord_vec, velocity_vec, acceleration_vec =
            #velocity_verlet_integration(coord_vec, velocity_vec, acceleration_vec)
            #langevin_integration(coord_vec, velocity_vec, acceleration_vec)
            # BAOAB_langevin_integration(coord_vec, velocity_vec, acceleration_vec)
            g_BAOAB_langevin_integration(coord_vec, velocity_vec, acceleration_vec)

        # for dump
        if mod(step_idx, dump_step) == 0
            println("step ", step_idx)
            push!(frame_vec_for_dump, coord_vec)
            push!(vel_vec_for_dump, velocity_vec)
            push!(energy_vec_for_dump, calculate_energy(coord_vec, velocity_vec, mass_vec))
        end
    end

    # output part
    open("trajectory.xyz", "w") do os
        for (idx, frame) in enumerate(frame_vec_for_dump)
            println(os, particle_num)
            println(os, "step = ", idx * dump_step)
            for atom_idx in 1:2:size(frame, 2)
                core = frame[:, atom_idx]
                patch = frame[:, atom_idx + 1]
                @printf(os, "CORE  %11.8f %11.8f %11.8f\n", core[1], core[2], core[3])
                @printf(os, "PATCH %11.8f %11.8f %11.8f\n", patch[1], patch[2], patch[3])
            end
        end
    end

    open("velocity.xyz", "w") do os
        for (idx, frame) in enumerate(vel_vec_for_dump)
            println(os, particle_num)
            println(os, "step = ", idx * dump_step)
            for (idx, atom) in enumerate(eachcol(frame))
                @printf(os, "Particle%d %11.8f %11.8f %11.8f\n", idx, atom[1], atom[2], atom[3])
            end
        end
    end

    open("energy.dat", "w") do os
        for energy in energy_vec_for_dump
            println(os, energy)
        end
    end
end

main()
close(log_file)

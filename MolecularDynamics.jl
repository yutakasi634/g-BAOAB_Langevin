using Printf
using Random
using Distributions
using LinearAlgebra

# simulation meta information
const time_step = 0.1
const total_step = 50000
const dump_step = 100

# physical constants
const kB = 0.0019872 # kcal/ mol K

# system paraneters
const particle_num  = 50
const lennard_jones_eps = 0.6
const lennard_jones_sigma = 2.0
const mass_range             = (80.0   ,120.0)
const gamma = 0.01
const temperature = 300.0

# random initial setting parameters
const initial_position_range = (-15.0, 15.0)
const initial_velocity_range = (-0.1, 0.1)

# box parameters
const box_side_length        = 40.0
const box_eps                = 0.6
const box_sigma              = 4.0

# random number generator paramters
const rng = MersenneTwister(1234)

# pre calculation
const time_step2 = time_step * time_step
const one_minus_gammah_over2 = 1 - gamma * time_step * 0.5
const mass_vec     = rand(Uniform(mass_range...), 1, particle_num)
const inv_mass_vec = 1 ./ mass_vec
const sqrt_inv_mass_vec = sqrt.(inv_mass_vec)
const noise_coef_vec = sqrt.(2gamma * kB * temperature / time_step .* inv_mass_vec)
const baoab_c_1 = ℯ^(-gamma * time_step)
const baoab_c_3 = sqrt(kB * temperature * (1 - baoab_c_1^2))

log_file = open("logfile.log", "w")

include("force_energy_calculation.jl")
include("integrations.jl")

function main()
    println("main start")
    # random system initilization
    coord_vec    = rand(Uniform(initial_position_range...), 3, particle_num)
    velocity_vec = zeros(3, particle_num)

    # pre preparation
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
            BAOAB_langevin_integration(coord_vec, velocity_vec, acceleration_vec)

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
            for (idx, atom) in enumerate(eachcol(frame))
                @printf(os, "Particle%d %11.8f %11.8f %11.8f\n", idx, atom[1], atom[2], atom[3])
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

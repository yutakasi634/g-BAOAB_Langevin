using Printf
using Random
using Distributions
using LinearAlgebra

# simulation meta information
const time_step = 0.1
const total_step = 100000
const dump_step = 100

# physical constants
const kB = 0.0019872 # kcal/ mol K

# system paraneters
const particle_num  = 5
const lennard_jones_eps = 0.6
const lennard_jones_sigma = 2.0
const initial_position_range = (-8.0 ,8.0)
const initial_velocity_range = (-0.1 ,0.1)
const mass_range             = (80.0   ,120.0)
const gamma = 0.01
const temperature = 300.0

# box parameters
const box_side_length        = 25.0
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

function lennard_jones(eps::Float64, sig::Float64, r::Float64)::Float64
    sig_r_6 = (sig / r)^6
    sig_r_12 = sig_r_6^2
    4eps * (sig_r_12 - sig_r_6)
end

function dev_lennard_jones(eps::Float64, sig::Float64, r::Float64)::Float64
    rinv = 1.0 / r
    sig_r_6 = (sig * rinv)^6
    sig_r_12 = sig_r_6^2
    24.0 * eps * rinv * (sig_r_6 - 2.0 * sig_r_12)
end

function excluded_volume(eps::Float64, sig::Float64, r::Float64)::Float64
    eps * (sig / r)^12
end

function dev_excluded_volume(eps::Float64, sig::Float64, r::Float64)::Float64
    rinv = 1.0 / r
    -12.0 * eps * rinv * (sig * rinv)^12
end

function calculate_energy(coord_vec::Array{Float64, 2}, velocity_vec::Array{Float64, 2},
                          mass_vec::Array{Float64, 2})::Float64
    total_energy = 0
    # lennard jones part
    for i in 1:particle_num - 1
        for j in i + 1:particle_num
            distance = norm(coord_vec[:,i] - coord_vec[:,j])
            println(log_file, "distance between $(i), $(j) ", distance)
            total_energy += lennard_jones(lennard_jones_eps, lennard_jones_sigma, distance)
        end
    end

    println(log_file, "lennard jones energy ", total_energy)

    # box potential part
    box_side_coord = box_side_length * 0.5
    total_energy +=
        sum(coord -> excluded_volume(box_eps, box_sigma, box_side_coord - abs(coord)), coord_vec)
    println(log_file, "total energy ", total_energy)

    # physical energy
    total_energy += sum(0.5 * velocity_vec .* velocity_vec .* mass_vec)

    total_energy
end

function calculate_force(coord_vec::Array{Float64, 2})::Array{Float64, 2}
    res_force_vec = zeros(3, particle_num)
    # lennard jones part
    for i in 1:particle_num - 1
        for j in i+1:particle_num
            dist_vec = coord_vec[:,j] - coord_vec[:,i]
            distance = norm(dist_vec)
            lennard_force_vec = dev_lennard_jones(lennard_jones_eps, lennard_jones_sigma, distance) * dist_vec / distance
            res_force_vec[:,i] += lennard_force_vec
            res_force_vec[:,j] -= lennard_force_vec
        end
    end

    #println(log_file, "lennard jones force ", res_force_vec)

    # box potential part
    box_side_coord = box_side_length * 0.5
    box_force_vec = map(coord_vec) do coord
        dev_excluded_volume(box_eps, box_sigma, box_side_coord - abs(coord)) * sign(coord)
    end
    #println(log_file, "box force ", box_force_vec)
    res_force_vec += box_force_vec

    #println(log_file, "total force ", res_force_vec)
    res_force_vec
end

function velocity_verlet_integration(coord_vec::Array{Float64, 2}, vel_vec::Array{Float64, 2},
                                     acceleration_vec::Array{Float64, 2})::Tuple{Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}
    new_coord_vec =
        coord_vec + time_step * vel_vec + time_step2 * 0.5 * acceleration_vec
    new_acceleration_vec = calculate_force(new_coord_vec) .* inv_mass_vec
    new_vel_vec = vel_vec + time_step * 0.5 * (acceleration_vec + new_acceleration_vec)

    (new_coord_vec, new_vel_vec, new_acceleration_vec)
end

function langevin_integration(coord_vec::Array{Float64, 2}, vel_vec::Array{Float64, 2},
                              acceleration_vec::Array{Float64, 2})::Tuple{Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}
    new_coord_vec =
        coord_vec + vel_vec * time_step * one_minus_gammah_over2 +
        time_step2 * 0.5 * acceleration_vec
    new_acceleration_vec =
        calculate_force(new_coord_vec) .* inv_mass_vec + noise_coef_vec .*
        randn(rng, (3, particle_num))
    new_vel_vec =
        one_minus_gammah_over2 * (one_minus_gammah_over2 + (gamma * time_step * 0.5)^2) .* vel_vec +
        0.5 * time_step * one_minus_gammah_over2 * (acceleration_vec + new_acceleration_vec)
    (new_coord_vec, new_vel_vec, new_acceleration_vec)
end

function BAOAB_langevin_integration(coord_vec::Array{Float64, 2}, vel_vec::Array{Float64, 2},
                                    acceleration_vec::Array{Float64, 2})::Tuple{Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}
    temp_vel_vec         = vel_vec + time_step * 0.5 .* acceleration_vec
    temp_coord_vec       = coord_vec + time_step * 0.5 .* temp_vel_vec
    temp_hat_vel_vec     =
        baoab_c_1 .* temp_vel_vec + baoab_c_3 .* sqrt_inv_mass_vec .* randn(rng, (3, particle_num))
    new_coord_vec        = temp_coord_vec + time_step * 0.5 .* temp_hat_vel_vec
    new_acceleration_vec = calculate_force(new_coord_vec) .* inv_mass_vec
    new_vel_vec          = temp_hat_vel_vec + time_step * 0.5 .* new_acceleration_vec
    (new_coord_vec, new_vel_vec, new_acceleration_vec)
end

function main()
    println("main start")
    # system initilization
    coord_vec    = rand(Uniform(initial_position_range...), 3, particle_num)
    #velocity_vec = rand(Uniform(initial_velocity_range...), 3, particle_num)
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

using Printf
using Random
using Distributions
using LinearAlgebra

const particle_num  = 5
const time_step = 0.01
const total_step = 10000
const dump_step = 10
const lennard_jones_eps = 0.6
const lennard_jones_sigma = 2.0
const initial_position_range = (-10.0 ,10.0)
const initial_velocity_range = (-10.0 ,10.0)
const mass_range             = (100.0   ,101.0)
const box_side_length        = 25.0
const box_eps                = 0.6
const box_sigma              = 4.0

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
                                     force_vec::Array{Float64, 2}, inv_mass_vec::Array{Float64, 2})::Tuple{Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}
    new_coord_vec =
        coord_vec + time_step * vel_vec + time_step * time_step * force_vec .* inv_mass_vec * 0.5
    new_force_vec = calculate_force(new_coord_vec)
    new_vel_vec = vel_vec + time_step * (force_vec + new_force_vec) .* inv_mass_vec * 0.5

    (new_coord_vec, new_vel_vec, new_force_vec)
end

function main()
    println("main start")
    # system initilization
    coord_vec    = rand(Uniform(initial_position_range...), 3, particle_num)
    velocity_vec = rand(Uniform(initial_velocity_range...), 3, particle_num)
    mass_vec     = rand(Uniform(mass_range...), 1, particle_num)

    # pre preparation
    inv_mass_vec = 1 ./ mass_vec
    frame_vec_for_dump = []
    energy_vec_for_dump = []
    push!(frame_vec_for_dump, coord_vec)

    # system integration
    force_vec        = calculate_force(coord_vec)
    # loop start
    for step_idx in 1:total_step
        coord_vec, velocity_vec, force_vec = velocity_verlet_integration(coord_vec, velocity_vec, force_vec, inv_mass_vec)

        # for dump
        if mod(step_idx, dump_step) == 0
            println("step ", step_idx)
            push!(frame_vec_for_dump, coord_vec)
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

    open("energy.dat", "w") do os
        for energy in energy_vec_for_dump
            println(os, energy)
        end
    end
end

main()
close(log_file)

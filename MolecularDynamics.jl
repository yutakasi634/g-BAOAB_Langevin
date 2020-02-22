using Random
using Distributions
using LinearAlgebra

const particle_num  = 2
const time_step = 0.01
const total_step = 300
const dump_step = 1
const lennard_jones_eps = 0.6
const initial_position_range = (-10.0 ,10.0)
const initial_velocity_range = (-10.0 ,10.0)
const mass_range             = (100.0   ,101.0)
const box_side_length        = 30.0
const box_eps                = 0.6
const box_sigma              = 0.5

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

function dev_excluded_volume(eps::Float64, sig::Float64, r::Float64)::Float64
    rinv = 1.0 / r
    -12.0 * eps * rinv * (sig * rinv)^12
end

function calculate_force(coord_vec::Array{Float64, 2})::Array{Float64, 2}
    res_force_vec = zeros(3, particle_num)
    # lennard jones part
    for i in 1:particle_num - 1
        for j in i+1:particle_num
            dist_vec = coord_vec[:,j] - coord_vec[:,i]
            distance = norm(dist_vec)
            lennard_force_vec = dev_lennard_jones(lennard_jones_eps, 0.1, distance) * dist_vec / distance
            res_force_vec[:,i] += lennard_force_vec
            res_force_vec[:,j] -= lennard_force_vec
        end
    end

    # box potential part
    box_side_coord = box_side_length * 0.5
    box_force_vec = map(coord_vec) do coord
        dev_excluded_volume(box_eps, box_sigma, box_side_coord - abs(coord)) * sign(coord)
    end
    res_force_vec += box_force_vec

    res_force_vec
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
    push!(frame_vec_for_dump, coord_vec)

    # system integration
    force_vec        = calculate_force(coord_vec)
    # loop start
    for step_idx in 1:total_step
        coord_vec = coord_vec + time_step * velocity_vec + time_step * time_step * force_vec .* inv_mass_vec * 0.5
        new_force_vec = calculate_force(coord_vec)
        velocity_vec = velocity_vec + time_step * (force_vec + new_force_vec) * 0.5

        force_vec = new_force_vec

        if mod(step_idx, dump_step) == 0
            println("step ", step_idx)
            push!(frame_vec_for_dump, coord_vec)
        end
    end

    # output part
    open("trajectory.xyz", "w") do os
        for (idx, frame) in enumerate(frame_vec_for_dump)
            println(os, particle_num)
            println(os, "step = ", idx * dump_step)
            for (idx, atom) in enumerate(eachcol(frame))
                println(os,"Particle" ,idx ," " , atom[1], " ", atom[2], " ", atom[3])
            end
        end
    end
end

main()

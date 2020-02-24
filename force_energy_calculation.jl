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

function dev_harmonic(k::Float64, v0::Float64, v::Float64)::Float64
    2k * (v - v0)
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

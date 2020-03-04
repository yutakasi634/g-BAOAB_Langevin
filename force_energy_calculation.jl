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

function harmonic(k::Float64, v0::Float64, v::Float64)::Float64
    k * (v - v0)^2
end

function dev_harmonic(k::Float64, v0::Float64, v::Float64)::Float64
    2k * (v - v0)
end

function inverse_power(eps::Float64, sig::Float64, r::Float64, n::Int64)::Float64
    eps * (sig / r)^n
end

function dev_inverse_power(eps::Float64, sig::Float64, r::Float64, n::Int64)::Float64
    inv_r = 1.0 / r
    -n * eps * inv_r * (sig * inv_r)^n
end

function calculate_energy(coord_vec::Array{Float64, 2}, velocity_vec::Array{Float64, 2},
                          mass_vec::Array{Float64, 2})::Float64
    total_energy = 0
    # inverse power part
    for first_idx in 1:2:patch_particle_num - 1
        for second_idx in first_idx + 2:2:patch_particle_num
            dist_vec = coord_vec[:, second_idx] - coord_vec[:, first_idx]
            distance = norm(dist_vec)
            total_energy +=
                inverse_power(inverse_power_coef, core_patch_dist * 2, distance, inverse_power_n)
        end
    end

    # println(log_file, "lennard jones energy ", total_energy)

    # harmonic bond part
    for patch_particle_idx in 1:patch_particle_num
        dist_vec = coord_vec[:, patch_particle_idx * 2 - 1] - coord_vec[:, patch_particle_idx * 2]
        distance = norm(dist_vec)
        total_energy += harmonic(core_patch_bond_coef, core_patch_dist, distance)
    end

    # box potential part
    box_side_coord = box_side_length * 0.5
    total_energy +=
        sum(coord -> excluded_volume(box_eps, box_sigma, box_side_coord - abs(coord)),
            coord_vec[:, 1:2:particle_num])
    # println(log_file, "total energy ", total_energy)

    # physical energy
    total_energy += sum(0.5 * velocity_vec .* velocity_vec .* mass_vec)

    total_energy
end

function calculate_force(coord_vec::Array{Float64, 2})::Array{Float64, 2}
    res_force_vec = zeros(3, particle_num)
    # inverse power part
    for first_idx in 1:2:patch_particle_num - 1
        for second_idx in first_idx + 2:2:patch_particle_num
            dist_vec = coord_vec[:, second_idx] - coord_vec[:, first_idx]
            distance = norm(dist_vec)
            inverse_power_force_vec =
                dev_inverse_power(inverse_power_coef, core_patch_dist * 2,
                                  distance, inverse_power_n) * dist_vec /distance
            res_force_vec[:, first_idx] += inverse_power_force_vec
            res_force_vec[:, second_idx] -= inverse_power_force_vec
        end
    end

    #println(log_file, "inverse power force ", res_force_vec)

    # harmonic bond part
    # for patch_particle_idx in 1:patch_particle_num
    #     core_idx = patch_particle_idx * 2 - 1
    #     patch_idx = core_idx + 1
    #     dist_vec = coord_vec[:, patch_idx] - coord_vec[:, core_idx]
    #     distance = norm(dist_vec)
    #     harmonic_bond_force_vec =
    #         dev_harmonic(core_patch_bond_coef, core_patch_dist, distance) * dist_vec / distance
    #     res_force_vec[:, core_idx]  += harmonic_bond_force_vec
    #     res_force_vec[:, patch_idx] -= harmonic_bond_force_vec
    # end

    # box potential part
    box_side_coord = box_side_length * 0.5
    box_force_vec = map(coord_vec[:, 1:2:particle_num]) do coord
        dev_excluded_volume(box_eps, box_sigma, box_side_coord - abs(coord)) * sign(coord)
    end
    #println(log_file, "box force ", box_force_vec)
    res_force_vec[:,1:2:particle_num] += box_force_vec

    #println(log_file, "total force ", res_force_vec)
    res_force_vec
end

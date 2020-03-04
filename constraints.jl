function rattle4velocity!(coord_vec::Array{Float64, 2}, vel_vec::Array{Float64, 2}, time_step::Float64)
    for iter_idx in 1:rattle_max_iter
        correction_occur = false
        for core_idx in 1:2:particle_num
            patch_idx = core_idx + 1
            dist_vec = coord_vec[:, patch_idx] - coord_vec[:, core_idx]
            vel_diff_vec =  vel_vec[:, patch_idx] - vel_vec[:, core_idx]
            product_dist_vel_diff = dot(dist_vec, vel_diff_vec)
            lambda =
                product_dist_vel_diff / ((inv_mass_vec[core_idx] + inv_mass_vec[patch_idx]) * core_patch_dist2)
            if abs(lambda) > rattle_tolerance / time_step
                correction_occur = true
                correction_vec = lambda * dist_vec
                vel_vec[:, core_idx]  += correction_vec * inv_mass_vec[core_idx]
                vel_vec[:, patch_idx] -= correction_vec * inv_mass_vec[patch_idx]
            end
        end

        if correction_occur == false
            println(log_file, "rattle for velocity: break iter_idx ", iter_idx)
            break
        end

        if iter_idx == rattle_max_iter
            error("rattle iteration reach to max iteration step.")
        end
    end
end

function rattle4coordinate!(coord_vec::Array{Float64, 2}, old_coord_vec::Array{Float64, 2},
                            vel_vec::Array{Float64, 2}, time_step::Float64)
    inv_time_step = 1.0 / time_step
    for iter_idx in 1:rattle_max_iter
        correction_occur = false
        for core_idx in 1:2:particle_num
            patch_idx = core_idx + 1
            dist_vec = coord_vec[:, patch_idx] - coord_vec[:, core_idx]
            dist2 = dot(dist_vec, dist_vec)
            length_missmatch2 =  core_patch_dist2 - dist2
            if abs(length_missmatch2) > rattle_tolerance
                correction_occur = true
                old_dist_vec = old_coord_vec[:, patch_idx] - old_coord_vec[:, core_idx]
                product_old_new_dist_vec = dot(dist_vec, old_dist_vec)
                lambda =
                    0.5 * length_missmatch2 /
                    ((inv_mass_vec[core_idx] + inv_mass_vec[patch_idx]) * product_old_new_dist_vec)
                correction_force_vec = lambda * old_dist_vec
                core_correction_vec  = correction_force_vec * inv_mass_vec[core_idx]
                patch_correction_vec = correction_force_vec * inv_mass_vec[patch_idx]
                coord_vec[:, core_idx]  -= core_correction_vec
                coord_vec[:, patch_idx] += patch_correction_vec
                vel_vec[:, core_idx]    -= core_correction_vec * inv_time_step
                vel_vec[:, patch_idx]   += patch_correction_vec * inv_time_step
            end
        end

        if correction_occur == false
            println(log_file, "rattle for coordinate: break iter_idx ", iter_idx)
            break
        end

        if iter_idx == rattle_max_iter
            error("rattle iteration reach to max iteration step.")
        end

    end
end

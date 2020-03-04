function velocity_verlet_integration(coord_vec::Array{Float64, 2}, vel_vec::Array{Float64, 2},
                                     acceleration_vec::Array{Float64, 2})::Tuple{Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}
    new_coord_vec =
        coord_vec + time_step * vel_vec + time_step2 * 0.5 * acceleration_vec
    new_acceleration_vec = calculate_force(new_coord_vec) .* inv_mass_vec
    new_vel_vec = vel_vec + half_time_step * (acceleration_vec + new_acceleration_vec)

    (new_coord_vec, new_vel_vec, new_acceleration_vec)
end

function langevin_integration(coord_vec::Array{Float64, 2}, vel_vec::Array{Float64, 2},
                              acceleration_vec::Array{Float64, 2})::Tuple{Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}
    new_coord_vec =
        coord_vec + vel_vec * time_step .* one_minus_gammah_over2_vec +
        time_step2 * 0.5 * acceleration_vec
    new_acceleration_vec =
        calculate_force(new_coord_vec) .* inv_mass_vec + noise_coef_vec .*
        randn(rng, (3, particle_num))
    new_vel_vec =
        one_minus_gammah_over2_vec .* (one_minus_gammah_over2_vec + (gamma_vec * half_time_step).^2) .* vel_vec +
        0.5 * time_step * one_minus_gammah_over2_vec .* (acceleration_vec + new_acceleration_vec)
    (new_coord_vec, new_vel_vec, new_acceleration_vec)
end

function BAOAB_langevin_integration(coord_vec::Array{Float64, 2}, vel_vec::Array{Float64, 2},
                                    acceleration_vec::Array{Float64, 2})::Tuple{Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}
    temp_vel_vec         = vel_vec + half_time_step .* acceleration_vec # B step
    temp_coord_vec       = coord_vec + half_time_step .* temp_vel_vec # A step
    temp_hat_vel_vec     =
        baoab_c_1_vec .* temp_vel_vec + baoab_c_3_vec .* sqrt_inv_mass_vec .* randn(rng, (3, particle_num)) # O step
    new_coord_vec        = temp_coord_vec + half_time_step .* temp_hat_vel_vec # A step
    new_acceleration_vec = calculate_force(new_coord_vec) .* inv_mass_vec
    new_vel_vec          = temp_hat_vel_vec + half_time_step .* new_acceleration_vec # B step
    (new_coord_vec, new_vel_vec, new_acceleration_vec)
end

"""
This correspond to g-BAOAB langevin integration of 1 time rattle integration case.
"""
function g_BAOAB_langevin_integration(coord_vec::Array{Float64, 2}, vel_vec::Array{Float64, 2},
                                      acceleration_vec::Array{Float64, 2})::Tuple{Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}

    # B step
    temp_vel_vec         = vel_vec + half_time_step .* acceleration_vec
    rattle4velocity!(coord_vec, temp_vel_vec, time_step) # correct temp_vel_vec

    # A step
    temp_coord_vec       = coord_vec + half_time_step .* temp_vel_vec
    # correct temp_coord_vec (and temp_vel_vec)
    rattle4coordinate!(temp_coord_vec, coord_vec, temp_vel_vec, half_time_step)
    rattle4velocity!(temp_coord_vec, temp_vel_vec, half_time_step) # correct temp_vel_vec

    # O step
    temp_hat_vel_vec     =
        baoab_c_1_vec .* temp_vel_vec + baoab_c_3_vec .* sqrt_inv_mass_vec .* randn(rng, (3, particle_num))
    rattle4velocity!(temp_coord_vec, temp_hat_vel_vec, time_step) # correct temp_hat_vel_vec

    # A step
    new_coord_vec   = temp_coord_vec + half_time_step .* temp_hat_vel_vec
    # correct new_coord_vec (and temp_hat_vel_vec)
    rattle4coordinate!(new_coord_vec, temp_coord_vec, temp_hat_vel_vec, half_time_step)
    rattle4velocity!(new_coord_vec, temp_hat_vel_vec, half_time_step) # correct temp_hat_vel_vec

    # B step
    new_acceleration_vec = calculate_force(new_coord_vec) .* inv_mass_vec
    new_vel_vec = temp_hat_vel_vec + half_time_step .* new_acceleration_vec
    rattle4velocity!(new_coord_vec, new_vel_vec, time_step) # correct new_vel_vec

    (new_coord_vec, new_vel_vec, new_acceleration_vec)
end

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

function ude_dynamics_full!(
    du,
    u,
    p::AbstractVector{T},
    t,
    parameter_bounds,
    nn_model,
    st_nn,
    state_idx_nn,
) where {T}

    J_0, k_1, k_2, k_3, k_4, k_6, k_ex, kappa, K_1, N, A, phi = [
        inverse_transform(
            p[i],
            lb = parameter_bounds["lowerBound"][i],
            ub = parameter_bounds["upperBound"][i],
            par_transform = parameter_bounds["parameter_transform"][i],
        ) for i = 1:length(parameter_bounds["lowerBound"])
    ]

    S_1, S_2, S_3, S_4, N_2, A_3, S_4_ex = @inbounds u[1:7]

    h_1 = 1 + (A_3 / K_1)^4

    # mechanistic terms
    du_mech = zeros(T, 7)
    du_mech[1] = J_0 - k_1 * S_1 * A_3 / h_1
    du_mech[2] = 2 * (k_1 * S_1 * A_3) / h_1 - k_2 * S_2 * (N - N_2) - k_6 * S_2 * N_2
    du_mech[3] = k_2 * S_2 * (N - N_2) - k_3 * S_3 * (A - A_3)
    du_mech[4] = k_3 * S_3 * (A - A_3) - k_4 * S_4 * N_2 - kappa * (S_4 - S_4_ex)
    du_mech[5] = k_2 * S_2 * (N - N_2) - k_4 * S_4 * N_2 - k_6 * S_2 * N_2
    du_mech[6] = -2 * (k_1 * S_1 * A_3) / h_1 + 2 * k_3 * S_3 * (A - A_3) # - k_5*A_3 
    du_mech[7] = phi * kappa * (S_4 - S_4_ex) - k_ex * S_4_ex

    # NN component
    state_idx_nn_in, state_idx_nn_out = state_idx_nn
    du_nn = nn_model(u[state_idx_nn_in], p.nn, st_nn)[1]
    # zero contribution to all states but state_idx_nn_out
    du_nn_full = zeros(T, 7)
    du_nn_full[state_idx_nn_out] = du_nn

    # combined dynamics
    for i = 1:7
        du[i] = du_mech[i] + du_nn_full[i]
    end
end;


"""
Mapping hidden states to observables
"""
function observable_mapping_full(state, p, nn_model, st_nn)
    NADH = @inbounds state[5, :]
    ATP = @inbounds state[6, :]

    return vcat(transpose(NADH), transpose(ATP))
end

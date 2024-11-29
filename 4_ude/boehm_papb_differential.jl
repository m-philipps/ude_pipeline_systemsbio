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
    state_length = 8

    Epo_degradation_BaF3, k_imp_homo, k_imp_hetero, k_exp_homo, k_exp_hetero, k_phos = [
        inverse_transform(
            p[i],
            lb = parameter_bounds["lowerBound"][i],
            ub = parameter_bounds["upperBound"][i],
            par_transform = parameter_bounds["parameter_transform"][i],
        ) for i = 1:length(parameter_bounds["lowerBound"])
    ]

    STAT5A, STAT5B, pApA, pApB, pBpB, nucpApA, nucpApB, nucpBpB =
        @inbounds u[1:state_length]

    BaF3_Epo = 1.25e-7 * exp(-Epo_degradation_BaF3 * t)
    v_nuc = 0.45
    v_cyt = 1.4

    # mechanistic terms
    du_mech = eltype(u).(zeros(state_length))
    # STAT5A
    du_mech[1] =
        -2 * BaF3_Epo * STAT5A^2 * k_phos - BaF3_Epo * STAT5A * STAT5B * k_phos +
        2 * v_nuc / v_cyt * k_exp_homo * nucpApA +
        v_nuc / v_cyt * k_exp_hetero * nucpApB
    # STAT5B
    du_mech[2] =
        -BaF3_Epo * STAT5A * STAT5B * k_phos - 2 * BaF3_Epo * STAT5B^2 * k_phos +
        v_nuc / v_cyt * k_exp_hetero * nucpApB +
        2 * v_nuc / v_cyt * k_exp_homo * nucpBpB
    # pApA
    du_mech[3] = BaF3_Epo * STAT5A^2 * k_phos - k_imp_homo * pApA
    # pApB
    # du_mech[4] = BaF3_Epo * STAT5A * STAT5B * k_phos - k_imp_hetero * pApB
    # pBpB 
    du_mech[5] = BaF3_Epo * STAT5B^2 * k_phos - k_imp_homo * pBpB
    # nucpApA
    du_mech[6] = v_cyt / v_nuc * k_imp_homo * pApA - k_exp_homo * nucpApA
    # nucpApB
    du_mech[7] = v_cyt / v_nuc * k_imp_hetero * pApB - k_exp_hetero * nucpApB
    # nucpBpB
    du_mech[8] = v_cyt / v_nuc * k_imp_homo * pBpB - k_exp_homo * nucpBpB

    # NN component
    state_idx_nn_in, state_idx_nn_out = state_idx_nn
    du_nn = nn_model(u[state_idx_nn_in], p.nn, st_nn)[1]
    # zero contribution to all states but state_idx_nn_out
    du_nn_full = eltype(u).(zeros(state_length))
    du_nn_full[state_idx_nn_out] = du_nn

    # combined dynamics
    for i = 1:state_length
        du[i] = du_mech[i] + du_nn_full[i]
    end
end;


"""
Mapping hidden states to observables
"""
function observable_mapping_full(state, p, nn_model, st_nn)
    specC17 = 0.107
    STAT5A, STAT5B, pApA, pApB, pBpB, _, _, _ = [state[x, :] for x in axes(state, 1)]

    pSTAT5A_rel =
        (100 * pApB .+ 200 * pApA * specC17) ./
        (pApB + STAT5A * specC17 .+ 2 * pApA * specC17)
    pSTAT5B_rel =
        -(100 * pApB .- 200 * pBpB * (specC17 - 1)) ./
        (STAT5B * (specC17 - 1) .- pApB + 2 * pBpB * (specC17 - 1))
    rSTAT5A_rel =
        (100 * pApB .+ 100 * STAT5A * specC17 .+ 200 * pApA * specC17) ./ (
            2 * pApB .+ STAT5A * specC17 .+ 2 * pApA * specC17 .- STAT5B * (specC17 - 1) .-
            2 * pBpB * (specC17 - 1)
        )

    return vcat(transpose(pSTAT5A_rel), transpose(pSTAT5B_rel), transpose(rSTAT5A_rel))
end

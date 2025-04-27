using Lux, LuxCUDA
using Flux: Conv
using CUDA
using Interpolations
using NPZ
using ImageTransformations
using Einsum

include("../modeling/image_encoder.jl")

#=
########################################################
# Test generali:
########################################################
=#

# Recupero i dati di input
attn = npzread("./manualTest/testFiles/attn.npy")
q = npzread("./manualTest/testFiles/q.npy")
rel_pos_h = npzread("./manualTest/testFiles/rel_pos_h.npy")
rel_pos_w = npzread("./manualTest/testFiles/rel_pos_w.npy")
parameters = npzread("./manualTest/testFiles/parameters.npy")

# Recupero i dati expected
expected_attn_with_rel_pos = 
    npzread("./manualTest/testFiles/expected_attn_with_rel_pos.npy")

#=
expected_rel_h = 
    npzread("./manualTest/testFiles/rel_h.npy")
expected_rel_w = 
    npzread("./manualTest/testFiles/rel_w.npy")
expected_r_q = 
    npzread("./manualTest/testFiles/expected_r_q.npy")
expected_Rh =
    npzread("./manualTest/testFiles/expected_Rh.npy")
expected_Rw =
    npzread("./manualTest/testFiles/expected_Rw.npy")
expected_attn_first_view = 
    npzread("./manualTest/testFiles/expected_attn_first_view.npy")
expected_rel_h_attn = 
    npzread("./manualTest/testFiles/rel_h_attn.npy")
expected_rel_w_attn = 
    npzread("./manualTest/testFiles/rel_w_attn.npy")
expected_attn_sum = 
    npzread("./manualTest/testFiles/attn_sum.npy")
expected_final_attn = npzread("./manualTest/testFiles/expected_final_attn.npy")
=#

# Estrazione dei parametri
B = parameters[1]
q_h = parameters[2]
q_w = parameters[3]
k_h = parameters[4]
k_w = parameters[5]
C = parameters[6]


q_size = (q_h, q_w)
k_size = (k_h, k_w)

attn_with_rel_pos = 
    add_decompose_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size)

println(
    "attn with rel pos corretto? ",
    isapprox(attn_with_rel_pos,
        expected_attn_with_rel_pos,
        rtol=1e-7
        )
    )

#display(attn_with_rel_pos)
#display(expected_attn_with_rel_pos)

#=
q_h, q_w = q_size
k_h, k_w = k_size

Rh = get_rel_pos(q_h, k_h, rel_pos_h)
Rw = get_rel_pos(q_w, k_w, rel_pos_w)


println(
    "Rh corretto? ",
    isapprox(Rh, expected_Rh, rtol=1e-5)
    )

println(
    "Rw corretto? ",
    isapprox(Rw, expected_Rw, rtol=1e-5)
    )


B, _, dim = size(q)


r_q = permutedims(reshape(q, B, q_h, q_w, dim), (1, 3, 2, 4))

println(
    "r_q corretto? ",
    isapprox(r_q, expected_r_q, rtol=1e-5)
)


# Viene ricreata la funzione einsum di Pythorch
# Einsum sta per Einstein Summation per moltiplicazioni e somme di tensori
@einsum rel_h[b,h,w,k] := r_q[b,h,w,c] * Rh[h,k,c]
@einsum rel_w[b,h,w,k] := r_q[b,h,w,c] * Rw[w,k,c]


println(
    "rel_h corretto? ",
    isapprox(rel_h, expected_rel_h, rtol=1e-5)
    )

println(
    "rel_w corretto? ",
    isapprox(rel_w, expected_rel_w, rtol=1e-5)
    )

# Vengono aggiunti gli embeddings posizionali alle attenzioni
# Prima reshape di attn
attn_reshaped = permutedims(
    reshape(attn, B, q_h, q_w, k_h, k_w),
    (1, 3, 2, 5, 4)
    )

println(
    "attn first view corretto? ",
    isapprox(attn_reshaped, expected_attn_first_view, rtol=1e-5)
    )
      

####### Mi trovo qui: 
#devo controllare come sono fatte rel_h e rel_w 
# che vengono sommate alla view di attn

rel_h_reshaped = reshape(rel_h, B, q_w, q_h, k_h, 1)
rel_w_reshaped = reshape(rel_w, B, q_w, q_h, 1, k_w)

println(
    "rel_h in attn corretto? ",
    isapprox(rel_h_reshaped, expected_rel_h_attn, rtol=1e-5)
    )

println(
    "rel_w in attn corretto? ",
    isapprox(rel_w_reshaped, expected_rel_w_attn, rtol=1e-5)
    )

    
# Aggiunta di rel_h e rel_w con broadcasting
attn_with_rel = 
    attn_reshaped .+ 
    reshape(rel_h, B, q_w, q_h, k_h, 1) .+ 
    reshape(rel_w, B, q_w, q_h, 1, k_w)

println(
    "attn sum corretto? ",
    isapprox(attn_with_rel, expected_attn_sum, rtol=1e-5)
    )


# Step 2: reshape unendo q_h*q_w e k_h*k_w
attn_final = reshape(
    permutedims(attn_with_rel, (1, 3, 2, 5, 4)),
    B,
    q_h*q_w,
    k_h*k_w
    )

println(
    "attn final corretto? ",
    isapprox(attn_final, expected_final_attn, rtol=1e-5)
    )
=#



#=
# Esecuzione della funzione
attn_with_rel_pos = 
    add_decompose_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size)

# Verifica del risultato
println(
    "Risultato finale corretto? ",
    isapprox(attn_with_rel_pos, expected_attn_with_rel_pos, rtol=1e-5)
    )
=#












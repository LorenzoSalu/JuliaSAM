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
parameters = 
    npzread("./manualTest/testFiles/parameters_windows_unpartition.npy")

windows = 
    npzread("./manualTest/testFiles/windows.npy")


# Recupero i dati expected
expected_output = 
    npzread("./manualTest/testFiles/expected_window_unpartition_output.npy")

expected_B = 
    npzread("./manualTest/testFiles/expected_B.npy")

expected_first_x = 
    npzread("./manualTest/testFiles/expected_first_x.npy")

expected_first_x_perm = 
    npzread("./manualTest/testFiles/expected_first_x_perm.npy")

expected_second_x = 
    npzread("./manualTest/testFiles/expected_second_x.npy")

# Estrazione dei parametri
window_size = parameters[1]
pad_hw = (parameters[2], parameters[3])
hw = (parameters[4], parameters[5])

Hp, Wp = pad_hw
H, W = hw

B = size(windows, 1) ÷ (Hp * Wp ÷ window_size ÷ window_size)

println("B corretto? ", isapprox(B, expected_B, rtol=1e-5))



first_x = permutedims(reshape(
    windows,
    B, 
    Hp ÷ window_size, 
    Wp ÷ window_size, 
    window_size,
    window_size,
    :
    ), (3, 2, 1, 4, 5, 6))

println(size(first_x))


println(
    "first x corretto? ",
    isapprox(first_x, expected_first_x, rtol=1e-4)
    )

first_x_perm = permutedims(first_x, (1, 2, 4, 3, 5, 6))
# In python viene usato .contiguous() su first_x_perm

println(
    "first x perm corretto? ",
    isapprox(first_x_perm, expected_first_x_perm, rtol=1e-4)
    )


second_x = reshape(
    permutedims(first_x_perm, (1, 3, 2, 5, 4, 6)),
    B,
    Hp,
    Wp,
    :
    )

println(
    "second x perm corretto? ",
    isapprox(second_x, expected_second_x, rtol=1e-4)
    )  

if Hp > H || Wp > W
    # In python viene usato .contiguous() su second_x
    second_x = second_x[:, 1:H, 1:W, :] 
end

println(
    "risultato senza funzione corretto? ",
    isapprox(second_x, expected_output, rtol=1e-4)
    ) 


output_function = window_unpartition(windows, window_size, pad_hw, hw)


println(
    "risultato funzione corretto? ",
    isapprox(output_function, expected_output, rtol=1e-4)
    ) 
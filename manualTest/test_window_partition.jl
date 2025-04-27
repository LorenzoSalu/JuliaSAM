using Lux, LuxCUDA
using Flux: Conv
using CUDA
using Interpolations
using NPZ
using ImageTransformations
using Einsum
using NNlib

include("../modeling/image_encoder.jl")

#=
########################################################
# Recupero dati:
########################################################
=#

# Recupero i dati di input
x = 
    npzread("./manualTest/testFiles/windows_partition_input_x.npy")

window_size = 
    npzread("./manualTest/testFiles/windows_partition_input_window_size.npy")[1]


# Recupero i dati expected

#=
expected_pad_h_pad_w = 
    npzread("./manualTest/testFiles/window_partition_expected_pad_h_pad_w.npy")

expected_first_x = 
    npzread("./manualTest/testFiles/window_partition_expected_first_x.npy")

expected_second_x = 
    npzread("./manualTest/testFiles/window_partition_expected_second_x.npy")

expected_third_x = 
    npzread("./manualTest/testFiles/window_partition_expected_third_x.npy")

=#

expected_windows = 
    npzread("./manualTest/testFiles/window_partition_expected_windows.npy")

expected_Hp_Wp = 
    npzread("./manualTest/testFiles/window_partition_expected_Hp_Wp.npy")



#=
########################################################
# Test generali:
########################################################
=#

#=
B, H, W, C = size(x)

pad_h = (window_size - H % window_size) % window_size
pad_w = (window_size - W % window_size) % window_size

println(
    "(pad_h, pad_w) corretto? ",
    isapprox([pad_h, pad_w], expected_pad_h_pad_w, rtol=1e-4)
    )


if pad_h > 0 || pad_w > 0
    x = pad_constant(x, (0, 0, 0, pad_h, 0, pad_w, 0, 0), 0)
end

println(
    "first x corretto? ",
    all(isapprox(x, expected_first_x, rtol=1e-4))
    )

Hp, Wp = H + pad_h, W + pad_w

println(
    "Hp, Wp corretto? ",
    isapprox([Hp, Wp], expected_Hp_Wp, rtol=1e-4)
    )

#=
println(
    "Second x corretto? ",
    isapprox(x, expected_second_x, rtol=1e-4)
    )
=#

x = permutedims(reshape(
    permutedims(x, (4, 3, 2, 1)),
    B, 
    Hp ÷ window_size,
    window_size,
    Wp ÷ window_size,
    window_size,
    C
    ), (1, 2, 3, 4, 5, 6))

windows = permutedims(reshape(
    permutedims(x, (4, 6, 3, 5, 1, 2)),
    :, window_size, window_size, C), (1, 3, 2, 4)
    )

println(
    "risultato codice corretto? ",
    isapprox(windows, expected_windows, rtol=1e-4)
    ) 
=#

output_windws, (output_Hp, output_Wp) = window_partition(x, window_size)

println(
    "risultato funzione corretto? ",
    isapprox(output_windws, expected_windows, rtol=1e-4)
    ) 

println(
    "Hp, Wp corretto? ",
    isapprox([output_Hp, output_Wp], expected_Hp_Wp, rtol=1e-4)
    )

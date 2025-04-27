using Lux, LuxCUDA
using Flux: Conv
using CUDA
using Interpolations
using NPZ
using ImageTransformations

include("../modeling/image_encoder.jl")

#=
# Simulazione di align_corners=False
function get_positions(in_len, out_len)
    scale = in_len / out_len
    return [scale * (i - 0.5) + 0.5 for i in 1:out_len]
end


function resize_rel_pos(rel_pos::AbstractArray{T,2}, max_rel_dist::Int) where T
       
    in_len = size(rel_pos, 1)
    cols = size(rel_pos, 2)

    x_new = get_positions(in_len, max_rel_dist)

    # Interpolatore lineare (OnGrid) e poi lo estendiamo fuori dai bordi con Flat()
    itp = interpolate(rel_pos, BSpline(Linear()), OnGrid())
    eitp = extrapolate(itp, Flat())

    # Applichiamo lo scale per accedere via coordinate reali
    sitp = scale(eitp, 1:in_len, 1:cols)

    # Eseguiamo l’interpolazione
    res = Float32[ sitp[x, j] for x in x_new, j in 1:cols ]

    return res
    
end



########################################################
# Test resize_rel_pos:
########################################################


q_size = 6
k_size = 4

rel_pos = Float64.(npzread("./manualTest/matrix.npy"))
expected_resized_rel_pos = npzread("./manualTest/expected_matrix.npy")


max_rel_dist = Int(2 * max(q_size, k_size) - 1)


if size(rel_pos)[1] != max_rel_dist
    rel_pos_resized = resize_rel_pos(rel_pos, max_rel_dist)
else
    rel_pos_resized = rel_pos
end


isclose = isapprox(
    rel_pos_resized,
    expected_resized_rel_pos,
    rtol=1e-6
    )


println(rel_pos_resized[:, 1])

println(
    expected_resized_rel_pos[:, 1]
    )

println("rel_pos_resized correto a 1e-6? ", isclose)



########################################################
# Test Ricerca coordinate:
########################################################


q_coords = Float32.(round.(
    reshape(0:q_size-1, :, 1) * max(k_size / q_size, 1.0),
    digits = 1
    ))
k_coords = Float32.(round.(
    reshape(0:k_size-1, 1, :) * max(q_size / k_size, 1.0),
    digits = 1
    ))

relative_coords = 
((q_coords .- k_coords) .+ ((k_size - 1) * max(q_size / k_size, 1.0))) .+ 1


println("q_coords: ", q_coords)
println("k_coords: ", k_coords)
println("relative_coords: ")
display(Int64.(trunc.(relative_coords)))


relative_coords = Int64.(trunc.(relative_coords))

result = zeros(Float32, (size(relative_coords)...), size(rel_pos_resized, 2))

for i in 1:size(rel_pos_resized)[2]
    for j in 1:size(relative_coords)[1]
        for k in 1:size(relative_coords)[2]
            result[j, k, i] = rel_pos_resized[relative_coords[j, k], i]
        end
    end
end



expected_result = npzread("./manualTest/get_rel_pos_result.npy")


#display(result)
display(expected_result[:,:,128])
display(result[:,:,128])


isclose = zeros(Bool, size(result)[3])

for i in 1:size(result)[3]
    isclose[i] = isapprox(result[:,:,i], expected_result[:,:,i], atol=1e-5)
end

println("Risultato finale corretto? ", all(isclose))

 result è formato nel seguente modo
risultano 128 matrici della dimensione di relative_coords
con (1, 1, 128) recupero il valore in (1, 1) nella matrice 128, es: 5
con il valore 5 recupero (5, 128) di rel_pos_resized
=#

########################################################
# Test Intera funzione:
########################################################


#=
max_rel_dist = Int(2 * max(q_size, k_size) - 1)


if size(rel_pos)[1] != max_rel_dist
    rel_pos_resized = resize_rel_pos(rel_pos, max_rel_dist)
else
    rel_pos_resized = rel_pos
end


q_coords = Float32.(round.(
    reshape(0:q_size-1, :, 1) * max(k_size / q_size, 1.0),
    digits = 1
    ))
k_coords = Float32.(round.(
    reshape(0:k_size-1, 1, :) * max(q_size / k_size, 1.0),
    digits = 1
    ))


relative_coords = 
((q_coords .- k_coords) .+ ((k_size - 1) * max(q_size / k_size, 1.0))) .+ 1

relative_coords = Int64.(trunc.(relative_coords))

output = zeros(Float32, (size(relative_coords)...), size(rel_pos_resized, 2))

for i in 1:size(rel_pos_resized)[2]
    for j in 1:size(relative_coords)[1]
        for k in 1:size(relative_coords)[2]
            output[j, k, i] = rel_pos_resized[relative_coords[j, k], i]
        end
    end
end



println(
    "resized_rel_pos corretto? ", 
    isapprox(rel_pos_resized, expected_resized_rel_pos, rtol=1e-6)
    )


println(
    "relative_coords corretto? ", 
    isapprox(relative_coords .- 1, expected_relative_coords, rtol=1e-6)
    )

=#

q_size = 10
k_size = 12

rel_pos = Float64.(npzread("./manualTest/testFiles/matrix.npy"))

expected_resized_rel_pos = 

npzread("./manualTest/testFiles/expected_matrix.npy")
expected_relative_coords = 

npzread("./manualTest/testFiles/expected_relative_coords.npy")
expected_output = 

npzread("./manualTest/testFiles/expected_get_rel_pos_output.npy")

output = get_rel_pos(q_size, k_size, rel_pos)

for i in 1:size(output)[3]
    isclose[i] = isapprox(output[:,:,i], expected_output[:,:,i], rtol=1e-5)
end

println("Risultato finale corretto? ", all(isclose))

display(output[:,:,128])
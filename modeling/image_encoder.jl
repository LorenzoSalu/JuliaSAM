using Flux
using Lux
using CUDA
using Interpolations
using TensorOperations
using Einsum


########################################################
# Attention: 
# divide la finestra a in partizioni più piccole 
########################################################

#=
struct Attention1
    dim::Int
    num_heads::Int
    scale::Float32
    qkv::Dense
    proj::Dense
    use_rel_pos::Bool
    rel_pos_h::Union{Nothing, Param{Matrix{Float32}}}
    rel_pos_w::Union{Nothing, Param{Matrix{Float32}}}
end

function Attention(
    dim::Int;
    num_heads::Int = 8,
    qkv_bias::Bool = true,
    use_rel_pos::Bool = false,
    rel_pos_zero_init::Bool = true,
    input_size::{Nothing, Tuple{Int, Int}} = nothing
    )
    
    head_dim = dim ÷ num_heads
    scale = 1 / sqrt(head_dim)

    qkv = Dense(dim, dim * 3, qkv_bias)
    proj = Dense(dim, dim)

    if use_rel_pos
        @assert input_size !== nothing 
        "Input size must be provided if using relative positional encoding."

        rel_pos_h = param(zeros(2 * input_size[1] - 1, head_dim))
        rel_pos_w = param(zeros(2 * input_size[2] - 1, head_dim))
    end
    
    return Attention(
        dim,
        num_heads, 
        scale, 
        qkv, 
        proj, 
        use_rel_pos, 
        rel_pos_h, 
        rel_pos_w
        )
end


function (m::Attention)(x::AbstractArray)
     B, H, W, _ = size(x)

     qkv = 

     return x
end
=#

########################################################
# window_partition: 
# divide la finestra a in partizioni più piccole 
########################################################

# Viene definita la funzione window_partition
function window_partition(
    x::AbstractArray,
    window_size::Int,
    )::Tuple{AbstractArray, Tuple{Int, Int}}

    # Recupero dimensioni
    B, H, W, C = size(x)

    # Calcolo del padding
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    # Aggiunta del padding se necessario
    if pad_h > 0 || pad_w > 0
        x = pad_constant(x, (0, 0, 0, pad_h, 0, pad_w, 0, 0), 0)
    end

    # Ricalcolo delle dimensioni
    Hp, Wp = H + pad_h, W + pad_w 

   # Calcolo delle finestre 
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

    # Ritorno dei valori ottenuti
    return windows, (Hp, Wp)
end



########################################################
# window_unpartition: 
# Ricostruisce la finestra a partire dalle partizioni 
########################################################

# Viene definita la funzione window_unpartition
function window_unpartition(
    windows::AbstractArray,
    window_size::Int,
    pad_hw::Tuple{Int, Int},
    hw::Tuple{Int, Int},
    )::AbstractArray

    # Recupero dimensioni
    Hp, Wp = pad_hw
    H, W = hw

    # Calcolo della dimesnione di batch
    B = size(windows, 1) ÷ (Hp * Wp ÷ window_size ÷ window_size)

    # Primo reshape
    # organizza le finestre in una griglia per ogni immagine nel batch.
    first_x = permutedims(reshape(
        windows,
        B, 
        Hp ÷ window_size, 
        Wp ÷ window_size, 
        window_size,
        window_size,
        :
        ), (3, 2, 1, 4, 5, 6))

    first_x_perm = permutedims(first_x, (1, 2, 4, 3, 5, 6))
    # In python viene usato .contiguous() su first_x_perm
        
    #Ricostruisce l'immagine con padding, 
    # combinando le finestre in un'unica sequenza
    second_x = reshape(
    permutedims(first_x_perm, (1, 3, 2, 5, 4, 6)),
    B,
    Hp,
    Wp,
    :
    )

    # Rimozione del padding se necessario
    if Hp > H || Wp > W
        # In python viene usato .contiguous() su second_x
        second_x = second_x[:, 1:H, 1:W, :] 
    end

    # Restituisce il risultato
    return second_x
end
    


########################################################
# add_decompose_rel_pos: definito per implementare 
# la “Decompose Relative Positional Embeddings”.
# sfruttando il concetto di attenzione 
########################################################

# Viene definita la funzione add_decompose_rel_pos
function add_decompose_rel_pos(
    attn::AbstractArray, # Mappa di attenzione
    q::AbstractArray, # Tensore che rappresenta la query
    rel_pos_h::AbstractArray, # Tensore embeddings posizionali per l'altezza
    rel_pos_w::AbstractArray, # Tensore embeddings posizionali per la larghezza
    q_size::Tuple{Int, Int}, # Dimensioni spaziali della query
    k_size::Tuple{Int, Int}, # Dimensioni spaziali della key
    )::AbstractArray
    
    # Recupero le dimensioni spaziali di q e k
    q_h, q_w = q_size
    k_h, k_w = k_size

    # Recupero degli embeddings posizionali per altezza e larghezza
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    # Recupero le dimensioni di q
    B, _, dim = size(q)


    r_q = permutedims(reshape(q, B, q_h, q_w, dim), (1, 3, 2, 4))

    # Viene ricreata la funzione einsum di Pythorch
    # Einsum sta per Einstein Summation per moltiplicazioni e somme di tensori
    # Calcola i contributi delle posizioni relative per altezza e larghezza
    @einsum rel_h[b,h,w,k] := r_q[b,h,w,c] * Rh[h,k,c]
    @einsum rel_w[b,h,w,k] := r_q[b,h,w,c] * Rw[w,k,c]

    # Reshape di attn
    attn_reshaped = permutedims(
        reshape(attn, B, q_h, q_w, k_h, k_w),
        (1, 3, 2, 5, 4)
        )

    # Vengono aggiunti gli embeddings posizionali alle attenzioni
    # Aggiunta di rel_h e rel_w con broadcasting
    attn_with_rel = 
        attn_reshaped .+ 
        reshape(rel_h, B, q_w, q_h, k_h, 1) .+ 
        reshape(rel_w, B, q_w, q_h, 1, k_w)

    # Reshape finale
    attn_final = reshape(
        permutedims(attn_with_rel, (1, 3, 2, 5, 4)),
        B,
        q_h*q_w,
        k_h*k_w
        )

    # Restituisce la mappa di attenzione sommati agli embeddings posizionali
    return attn_final
end



########################################################
# get_rel_pos: 
# Viene eseguito il calcolo delle posizioni relative tra tutte le q e k
# Tramite le posizioni relative viene calcolato il valore degli embeddings
########################################################

# Viene definita la funzione get_rel_pos
function get_rel_pos(
    q_size::Int,
    k_size::Int,
    rel_pos::AbstractArray, # (L, C)
    # (L : numero di posizioni relative possibili, 
    # C : dimensione del vettore embedding)
    )::AbstractArray

    # Calcolo distanza massima relativa
    max_rel_dist = Int(2 * max(q_size, k_size) - 1)

    # Controllo se c'è bisogno di interpolare gli embeddings
    # per ridimensionare la matrice
    if size(rel_pos)[1] != max_rel_dist
        rel_pos_resized = resize_rel_pos(rel_pos, max_rel_dist)
    else
        rel_pos_resized = rel_pos
    end
    
    # Calcolo le coordinate delle patch q
    q_coords = Float32.(round.(
    reshape(0:q_size-1, :, 1) * max(k_size / q_size, 1.0),
    digits = 1
    ))

    # Calcolo le coordinate delle patch k
    k_coords = Float32.(round.(
    reshape(0:k_size-1, 1, :) * max(q_size / k_size, 1.0),
    digits = 1
    ))

    # Calcolo la differenza tra le coordinate delle patch q e k
    relative_coords = 
    ((q_coords .- k_coords) .+ ((k_size - 1) * max(q_size / k_size, 1.0))) .+ 1

    # Tronca i valori per poterli utilizzare come indici
    relative_coords = Int64.(trunc.(relative_coords))

    # Inizializza il risultato come matrice di zeri
    result = zeros(
        Float32,
        (size(relative_coords)...),
        size(rel_pos_resized, 2)
        )

    # Viene creata una struttura 3D per memorizzare i valori degli embeddings
    for i in 1:size(rel_pos_resized)[2]
        for j in 1:size(relative_coords)[1]
            for k in 1:size(relative_coords)[2]
                result[j, k, i] = rel_pos_resized[relative_coords[j, k], i]
            end
        end
    end

    # Restituisce la matrice 3D dei valori degli embeddings
    return result
end



########################################################
# get_positions: 
# Calcola le posizioni di interpolazione
########################################################

# Viene definita la funzione get_positions
function get_positions(in_len, out_len)
    scale = in_len / out_len # Rapporto tra lunghezze di input e output

    # Viene ritornato un vettore di posizioni equispaziate
    # applicando le trasformazioni utili per ricreare quelle di SAM 
    return [scale * (i - 0.5) + 0.5 for i in 1:out_len]
end



########################################################
# resize_rel_pos: 
# Restituisce l' interpolazione ottuta per scalare la matrice
########################################################

# Viene definita la funzione resize_rel_pos
function resize_rel_pos(rel_pos::AbstractArray{T,2}, max_rel_dist::Int) where T
    
    in_len = size(rel_pos, 1) # Taglia prima dimensione di rel_pos
    cols = size(rel_pos, 2) # Taglia seconda dimensione di rel_pos

    x_new = get_positions(in_len, max_rel_dist) # Calcolo nuove coordinate

    # Viene creato l'oggetto per l'interpolazione lineare
    # per simulare il comportamento di interpolazione di SAM
    # É stato sfruttato OnGrid e Flat per i valori al di fuori dei bordi
    # Non bastava una semplice interpolazione lineare
    itp = interpolate(rel_pos, BSpline(Linear()), OnGrid())
    eitp = extrapolate(itp, Flat())

    # Viene applicato lo scale per accedere via coordinate reali
    sitp = scale(eitp, 1:in_len, 1:cols)

    # Viene eseguita l’interpolazione
    res = Float32[ sitp[x, j] for x in x_new, j in 1:cols ]

    # Restituisce la matrice interpolata
    return res
    
end



########################################################
# PatchEmbed: definito per implementare l'embedding delle patch dell'immagine
# con embedding si intende la rappresentazione delle patch come vettori numerici
# Serve per applicare una convoluzione 2D alle patch dell'immagine
########################################################

# Viene definito un tipo strutturato PatchEmbed
# Rappresenta un layer di embedding per le patch dell'immagine
struct PatchEmbed
    proj::Conv # proj è un campo di tipo Conv
end

# Costruttore per PatchEmbed con parametri di default
function PatchEmbed(;
    kernel_size::Tuple{Int, Int} = (16, 16),
    stride::Tuple{Int, Int} = (16, 16),
    padding::Tuple{Int, Int} = (0, 0),
    in_chans::Int = 3,
    embed_dim::Int = 768
    )
    # Crea un layer di convoluzione con i parametri specificati
    # Conv viene chiamato con:
    # function Conv(w::AbstractArray{T,N}, b = true, σ = identity;
    #    stride = 1, pad = 0, dilation = 1, groups = 1) where {T,N}
    proj = Conv(
        (kernel_size...,), # (Kernel_size..,) è una tupla di interi separati
        in_chans => embed_dim; 
        stride=stride, 
        pad=padding
        )
    
    # Restituisce un'istanza di PatchEmbed con il layer di convoluzione creato
    return PatchEmbed(proj)
end

# Viene definito il forward pass per PatchEmbed
# Input: (H, W, C, B)
# Output: (B, H', W', C)
function (m::PatchEmbed)(x::AbstractArray)
    # output da proj (H', W', C', B)
    y = m.proj(x) 
    # (H', W', C', B) -> (B, H', W', C')
    return permutedims(y, (4, 1, 2, 3))  
end
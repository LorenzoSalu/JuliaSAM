using Lux
using Lux: Dense
using Lux: Conv
using CUDA
using Interpolations
using TensorOperations
using Einsum
using Random
using NNlib
using Tullio
using LoopVectorization

include("../global_functions.jl")


"""
########################################################
# Block:
########################################################

	Block{N1, N2, A, M}

Defines a transformer block that combines multi-head self-attention, feedforward MLP, and two normalization layers. This design supports both global and window-based local attention mechanisms.

# Fields
- `norm1::N1`: First normalization layer (typically `LayerNorm`), applied before the attention module.
- `norm1_ps::NamedTuple`: Parameters associated with the first normalization layer.
- `norm1_st::NamedTuple`: State associated with the first normalization layer.
- `norm2::N2`: Second normalization layer, applied before the MLP block.
- `norm2_ps::NamedTuple`: Parameters associated with the second normalization layer.
- `norm2_st::NamedTuple`: State associated with the second normalization layer.
- `attn::A`: Multi-head self-attention module, which may use global or local attention depending on `window_size`.
- `mlp::M`: Feedforward MLP block applied after the second normalization.
- `window_size::Int`: If greater than `0`, local attention is applied using non-overlapping windows of this size. If `0`, global attention is used.
"""
struct Block{N1, N2, A, M}
	norm1::N1
	norm1_ps::NamedTuple
	norm1_st::NamedTuple
	norm2::N2
	norm2_ps::NamedTuple
	norm2_st::NamedTuple
	attn::A
	mlp::M
	window_size::Int
end


"""
	Block(; 
		dim::Int, 
		num_heads::Int, 
		mlp_ratio::Float32=4.0f0, 
		qkv_bias::Bool=true,
		norm_layer::Type=LayerNorm, 
		act_layer::Function=gelu_exact,
		use_rel_pos::Bool=false, 
		rel_pos_zero_init::Bool=true,
		window_size::Int=0, 
		input_size::Union{Nothing, Tuple{Int, Int}})

Represents a Transformer block used in an image encoder architecture

# Arguments
- `dim::Int`: Dimensionality of the input embeddings.
- `num_heads::Int`: Number of attention heads.
- `mlp_ratio::Float32`: Ratio between the MLP hidden dimension and the input dimension. Default is `4.0`.
- `qkv_bias::Bool`: If `true`, adds a bias term to the query, key, and value projections. Default is `true`.
- `norm_layer::Type`: The normalization layer constructor (e.g., `LayerNorm`). Must be compatible with Lux. Default is `LayerNorm`.
- `act_layer::Function`: Activation function used in the MLP block (e.g., `gelu_exact`). Default is `gelu_exact`.
- `use_rel_pos::Bool`: If `true`, enables relative positional embeddings in the attention module. Default is `false`.
- `rel_pos_zero_init::Bool`: If `true`, initializes relative positional parameters to zero. Default is `true`.
- `window_size::Int`: Size of the attention window. If `0`, global attention is used. Otherwise, local attention is applied over square windows of this size.
- `input_size::Union{Nothing, Tuple{Int, Int}}`: The spatial size of the input (height, width). Required when using global attention.


# Returns
A `Block` instance that encapsulates:
- Two normalization layers (`norm1`, `norm2`)
- A self-attention mechanism (`attn`)
- A feedforward MLP layer (`mlp`)
- Associated Lux parameters and states for the normalization layers
- A window size configuration for switching between global and local attention
"""
function Block(;
	dim::Int,
	num_heads::Int,
	mlp_ratio::Float32 = 4.0f0,
	qkv_bias::Bool = true,
	norm_layer::Type = LayerNorm,
	act_layer::Function = gelu_exact,
	use_rel_pos::Bool = false,
	rel_pos_zero_init::Bool = true,
	window_size::Int = 0,
	input_size::Union{Nothing, Tuple{Int, Int}},
)

	norm1 = norm_layer(dim, dims = 1, epsilon = 1.0f-6)
	norm1_ps, norm1_st = Lux.setup(Random.MersenneTwister(), norm1)

	norm2 = norm_layer(dim, dims = 1, epsilon = 1.0f-6)
	norm2_ps, norm2_st = Lux.setup(Random.MersenneTwister(), norm2)

	attn = Attention(
		dim,
		num_heads = num_heads,
		qkv_bias = qkv_bias,
		use_rel_pos = use_rel_pos,
		rel_pos_zero_init = rel_pos_zero_init,
		input_size = window_size == 0 ? input_size : (window_size, window_size),
	)

	mlp = MLPBlock(
		embedding_dim = dim,
		mlp_dim = Int.(dim * mlp_ratio),
		act = act_layer,
	)

	return Block(
		norm1,
		norm1_ps,
		norm1_st,
		norm2,
		norm2_ps,
		norm2_st,
		attn,
		mlp,
		window_size,
	)
end


"""
	(block::Block)(x::AbstractArray)

Applies the Transformer block to the input tensor `x`, performing normalization, attention, and MLP operations, optionally using windowed attention.

# Arguments
- `x::AbstractArray`: Input tensor of shape `(batch_size, height, width, channels)` or a compatible layout depending on the surrounding architecture.

# Returns
- The output tensor after applying the following operations:
  1. First normalization (`norm1`)
  2. Multi-head self-attention (`attn`)
  3. Residual connection
  4. Second normalization (`norm2`)
  5. MLP feedforward network
  6. Second residual connection

# Processing Details
- The input is first reshaped to match the expected format for normalization layers.
- If `window_size > 0`, the attention is applied within local windows using `window_partition` and `window_unpartition`; otherwise, global attention is used.
- The attention output is added back to the input (residual connection).
- A second normalization and MLP are applied, followed by another residual addition.
"""
function (self::Block)(x::AbstractArray)

	shortcut = x

	x_dim = size(x)
	x = sam_reshape(x, (:, x_dim[end]))'

	x, _ = self.norm1(
		x,
		self.norm1_ps,
		self.norm1_st,
	)

	x = sam_reshape(x', x_dim)

	if self.window_size > 0
		H, W = size(x, 2), size(x, 3)
		x, pad_hw = window_partition(x, self.window_size)
	end

	x = self.attn(x)

	if self.window_size > 0
		x = window_unpartition(x, self.window_size, pad_hw, (H, W))
	end

	x = shortcut .+ x

	x_dim = size(x)
	norm2_x = sam_reshape(x, (:, x_dim[end]))'

	norm2_x, _ = self.norm2(
		norm2_x,
		self.norm2_ps,
		self.norm2_st,
	)

	norm2_x = sam_reshape(norm2_x', x_dim)
	mlp_x = self.mlp(norm2_x)
	x = x .+ mlp_x

	return x
end

"""
########################################################
# Attention: 
########################################################

	struct Attention

Defines a multi-head self-attention mechanism, with optional relative positional encoding, commonly used in vision transformer blocks.

# Fields
- `dim::Int`: Input and output embedding dimension.
- `num_heads::Int`: Number of attention heads.
- `scale::Float64`: Scaling factor applied to dot-product attention scores.
- `qkv::Dense`: Linear projection that simultaneously computes queries, keys, and values.
- `qkv_ps::NamedTuple`: Parameters for the `qkv` projection layer.
- `qkv_st::NamedTuple`: State for the `qkv` projection layer.
- `proj::Dense`: Output projection layer applied after attention.
- `proj_ps::NamedTuple`: Parameters for the `proj` layer.
- `proj_st::NamedTuple`: State for the `proj` layer.
- `use_rel_pos::Bool`: Whether to use relative positional encodings.
- `rel_pos_h::Union{Nothing, Matrix{Float32}}`: Relative positional embeddings for height (optional).
- `rel_pos_w::Union{Nothing, Matrix{Float32}}`: Relative positional embeddings for width (optional).
"""
struct Attention
	dim::Int
	num_heads::Int
	scale::Float64
	qkv::Dense
	qkv_ps::NamedTuple
	qkv_st::NamedTuple
	proj::Dense
	proj_ps::NamedTuple
	proj_st::NamedTuple
	use_rel_pos::Bool
	rel_pos_h::Union{Nothing, Matrix{Float32}}
	rel_pos_w::Union{Nothing, Matrix{Float32}}
end


"""
	Attention(
		dim::Int; 
		num_heads::Int = 8, 
		qkv_bias::Bool = true,
		use_rel_pos::Bool = false, 
		rel_pos_zero_init::Bool = true,
		input_size::Union{Nothing, Tuple{Int, Int}} = nothing)

Creates a multi-head self-attention module, with optional 2D relative positional encodings for vision models.

# Arguments
- `dim::Int`: The input/output embedding dimension.
- `num_heads::Int`: Number of attention heads. Default is `8`.
- `qkv_bias::Bool`: If `true`, includes a learnable bias in the Q, K, V projections. Default is `true`.
- `use_rel_pos::Bool`: If `true`, enables 2D relative positional encoding. Default is `false`.
- `rel_pos_zero_init::Bool`: If `true`, initializes relative positional embeddings to zeros. Default is `true`.
- `input_size::Union{Nothing, Tuple{Int, Int}}`: Spatial dimensions (height, width) of the input, required if `use_rel_pos == true`.

# Returns
An `Attention` instance including:
- Learnable projection layers for QKV and output
- Optional 2D relative positional encodings
- Pre-initialized parameter and state tuples for Lux
"""
function Attention(
	dim::Int;
	num_heads::Int = 8,
	qkv_bias::Bool = true,
	use_rel_pos::Bool = false,
	rel_pos_zero_init::Bool = true,
	input_size::Union{Nothing, Tuple{Int, Int}} = nothing,
)

	head_dim = dim ÷ num_heads
	scale = 1 / sqrt(head_dim)

	qkv = 
        Dense(dim, dim * 3, use_bias = qkv_bias, init_weight = kaiming_uniform)
	qkv_ps, qkv_st = Lux.setup(Random.MersenneTwister(), qkv)

	proj = Dense(dim, dim, init_weight = kaiming_uniform)
	proj_ps, proj_st = Lux.setup(Random.MersenneTwister(), proj)

	rel_pos_h, rel_pos_w = nothing, nothing

	if use_rel_pos
		@assert input_size !== nothing
		"Input size must be provided if using relative positional encoding."

		rel_pos_h =
			zeros(Float32, 2 * input_size[1] - 1, head_dim)
		rel_pos_w =
			zeros(Float32, 2 * input_size[2] - 1, head_dim)
	end

	return Attention(
		dim,
		num_heads,
		scale,
		qkv,
		qkv_ps,
		qkv_st,
		proj,
		proj_ps,
		proj_st,
		use_rel_pos,
		rel_pos_h,
		rel_pos_w,
	)
end

"""
    (attn::Attention)(x::AbstractArray)

Applies the multi-head self-attention mechanism to the input tensor `x`, with optional 2D relative positional encoding.

# Arguments
- `x::AbstractArray`: Input tensor of shape `(B, H, W, C)`, where:
  - `B`: Batch size
  - `H`, `W`: Spatial dimensions (height and width)
  - `C`: Channel or embedding dimension

# Returns
- `x::AbstractArray`: Output tensor of shape `(B, H, W, C)` after attention and final projection.

# Processing Steps
1. The input is projected to queries, keys, and values using a shared linear layer (`qkv`).
2. The projections are reshaped and split across `num_heads` for multi-head attention.
3. Scaled dot-product attention scores are computed
4. If `use_rel_pos == true`, relative positional embeddings are added to the attention scores.
5. Softmax is applied along the last axis of the attention scores.
6. The attention output is computed and reshaped back to match the original spatial dimensions.
7. A final linear projection (`proj`) is applied to mix attention heads and produce the output embedding.
"""
function (self::Attention)(x::AbstractArray)

	B, H, W, _ = size(x)

	qkv, _ = self.qkv(
		permutedims(x, (4, 1, 2, 3)),
		self.qkv_ps,
		self.qkv_st,
	)

	qkv = sam_reshape(
		permutedims(qkv, (2, 3, 4, 1)),
		(B, H * W, 3, self.num_heads, :),
	)

	qkv = sam_reshape(
		permutedims(qkv, (3, 1, 4, 2, 5)),
		(3, B * self.num_heads, H * W, :),
	)

	q, k, v = collect(eachslice(qkv; dims = 1))

	scaled_q = q * self.scale

    attn = zeros(Float32, size(scaled_q, 1), size(scaled_q, 2), size(k, 2))
    for i in axes(scaled_q, 1)
        attn[i, :, :] = scaled_q[i, :, :] * k[i, :, :]'
    end
    
	if self.use_rel_pos
		attn = add_decompose_rel_pos(
			attn,
			q,
			self.rel_pos_h,
			self.rel_pos_w,
			(H, W),
			(H, W),
		)
	end

	attn = softmax(attn, dims = 3)

    attn_v = zeros(Float32, size(attn, 1), size(attn, 2), size(v, 3))
    for i in axes(attn, 1)
        attn_v[i, :, :] = attn[i, :, :] * v[i, :, :]
    end
     
	attn_v = sam_reshape(attn_v, (B, self.num_heads, H, W, :))
	attn_v = permutedims(attn_v, (1, 3, 4, 2, 5))

	x = sam_reshape(attn_v, (B, H, W, :))
	x = Float32.(x)
	x, _ = self.proj(
		permutedims(x, (4, 1, 2, 3)),
		self.proj_ps,
		self.proj_st,
	)
	x = permutedims(x, (2, 3, 4, 1))

	return x
end



"""
########################################################
# window_partition
########################################################

    window_partition(x::AbstractArray, window_size::Int) 
        -> Tuple{AbstractArray, Tuple{Int, Int}}

Partitions the spatial dimensions of the input tensor into non-overlapping windows of size `window_size x window_size`, optionally applying padding to handle edge cases.

# Arguments
- `x::AbstractArray`: Input tensor of shape `(B, H, W, C)`, where:
  - `B`: Batch size
  - `H`, `W`: Spatial dimensions (height and width)
  - `C`: Number of channels or embedding dimension
- `window_size::Int`: Size of each window along the spatial dimensions.

# Returns
- `windows::AbstractArray`: A tensor of shape `(N, window_size, window_size, C)`, where `N` is the total number of windows across all batches and spatial positions.
- `(Hp, Wp)::Tuple{Int, Int}`: The padded height and width of the input, after applying necessary padding for full window coverage.

# Processing Steps
1. Computes how much padding is needed along height and width to make `H` and `W` divisible by `window_size`.
2. Applies zero-padding to the input tensor if needed.
3. Reshapes the padded tensor to extract non-overlapping windows of the specified size.
4. Permutes and reshapes the tensor to flatten batch and spatial grid into a single window batch dimension.
"""
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
	x = sam_reshape(
		x,
		(B, Hp ÷ window_size, window_size, Wp ÷ window_size, window_size, C),
	)

	windows = sam_reshape(
		permutedims(x, (1, 2, 4, 3, 5, 6)),
		(:, window_size, window_size, C),
	)
	# Ritorno dei valori ottenuti
	return windows, (Hp, Wp)
end




"""
########################################################
# window_unpartition: 
########################################################

    window_unpartition(
        windows::AbstractArray,
        window_size::Int,
        pad_hw::Tuple{Int, Int},
        hw::Tuple{Int, Int},
    ) -> AbstractArray

Reconstructs the original (possibly padded) tensor from non-overlapping windowed partitions.

# Arguments
- `windows::AbstractArray`: Input tensor of shape `(N, window_size, window_size, C)`, where `N` is the number of windows and `C` is the channel dimension.
- `window_size::Int`: Size of the square windows along height and width.
- `pad_hw::Tuple{Int, Int}`: Tuple `(Hp, Wp)` specifying the padded height and width used during window partitioning.
- `hw::Tuple{Int, Int}`: Tuple `(H, W)` specifying the original (unpadded) height and width of the input.

# Returns
- `x::AbstractArray`: Reconstructed tensor of shape `(B, H, W, C)`, where `B` is the batch size.

# Processing Steps
1. Computes the batch size `B` from the number of windows and padded dimensions.
2. Reshapes the windows back into a grid structure per batch sample.
3. Permutes the dimensions to align window rows and columns correctly.
4. Merges the grid structure into a single tensor of shape `(B, Hp, Wp, C)`.
5. If padding was applied during partitioning, it is now removed to recover the original spatial dimensions `(H, W)`.
"""
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
	first_x = sam_reshape(
		windows,
		(B, Hp ÷ window_size, Wp ÷ window_size, window_size, window_size, :),
	)

	first_x_perm = permutedims(first_x, (1, 2, 4, 3, 5, 6))
	# In python viene usato .contiguous() su first_x_perm

	#Ricostruisce l'immagine con padding, 
	# combinando le finestre in un'unica sequenza
	second_x = sam_reshape(first_x_perm, (B, Hp, Wp, :))

	# Rimozione del padding se necessario
	if Hp > H || Wp > W
		# In python viene usato .contiguous() su second_x
		second_x = second_x[:, 1:H, 1:W, :]
	end

	# Restituisce il risultato
	return second_x
end


"""
########################################################
# add_decompose_rel_pos
########################################################

	add_decompose_rel_pos(
        attn::AbstractArray,
        q::AbstractArray,
        rel_pos_h::AbstractArray,
        rel_pos_w::AbstractArray,
        q_size::Tuple{Int, Int},
        k_size::Tuple{Int, Int},
    ) -> AbstractArray

Adds decomposed relative positional embeddings to the attention map along the height and width axes.

# Arguments
- `attn::AbstractArray`: Raw attention scores of shape `(B * num_heads, Q, K)`, where `Q = q_h * q_w` and `K = k_h * k_w`.
- `q::AbstractArray`: Query tensor of shape `(B * num_heads, Q, head_dim)`.
- `rel_pos_h::AbstractArray`: Relative positional embeddings for the height axis of shape `(2 * k_h - 1, head_dim)`.
- `rel_pos_w::AbstractArray`: Relative positional embeddings for the width axis of shape `(2 * k_w - 1, head_dim)`.
- `q_size::Tuple{Int, Int}`: Tuple `(q_h, q_w)` representing the spatial size of the query.
- `k_size::Tuple{Int, Int}`: Tuple `(k_h, k_w)` representing the spatial size of the key.

# Returns
- `attn_final::AbstractArray`: Modified attention map of shape `(B * num_heads, Q, K)` with relative positional bias added.

# Processing Steps
1. Retrieves positional embeddings using `get_rel_pos` for both height and width axes, based on query/key sizes.
2. Reshapes `q` into a 4D tensor `(B, q_h, q_w, head_dim)` to align with spatial structure.
3. Uses Einstein summation (`@einsum`) to compute positional attention components:
   - `rel_h`: Contribution from relative height embeddings.
   - `rel_w`: Contribution from relative width embeddings.
4. Reshapes the attention tensor to 5D `(B, q_h, q_w, k_h, k_w)` and adds `rel_h` and `rel_w` using broadcasting.
5. Flattens the spatial dimensions to return the final 2D attention map.
"""
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

	r_q = sam_reshape(q, (B, q_h, q_w, dim))

	# Viene ricreata la funzione einsum di Pythorch
	# Einsum sta per Einstein Summation per moltiplicazioni e somme di tensori
	# Calcola i contributi delle posizioni relative per altezza e larghezza
	@einsum rel_h[b, h, w, k] := r_q[b, h, w, c] * Rh[h, k, c]
	@einsum rel_w[b, h, w, k] := r_q[b, h, w, c] * Rw[w, k, c]

	# Reshape di attn
	attn_reshaped = sam_reshape(attn, (B, q_h, q_w, k_h, k_w))

	# Vengono aggiunti gli embeddings posizionali alle attenzioni
	# Aggiunta di rel_h e rel_w con broadcasting
	attn_with_rel =
		attn_reshaped .+
		reshape(rel_h, B, q_w, q_h, k_h, 1) .+
		reshape(rel_w, B, q_w, q_h, 1, k_w)

	# Reshape finale
	attn_final = sam_reshape(attn_with_rel, (B, q_h * q_w, k_h * k_w))

	# Restituisce la mappa di attenzione sommati agli embeddings posizionali
	return attn_final
end


"""
########################################################
# get_rel_pos
########################################################

    get_rel_pos(
        q_size::Int,
        k_size::Int,
        rel_pos::AbstractArray,
    ) -> AbstractArray

Retrieves the resized or indexed relative positional embeddings for a given query-key size configuration.

# Arguments
- `q_size::Int`: Spatial size of the query (number of patches along a dimension, e.g., height or width).
- `k_size::Int`: Spatial size of the key.
- `rel_pos::AbstractArray`: Relative positional embedding tensor of shape `(L, C)`, where:
    - `L` is the number of relative distances supported.
    - `C` is the embedding dimension (typically equal to head_dim).

# Returns
- `result::AbstractArray`: A tensor of shape `(q_size, k_size, C)` representing the selected (or interpolated) relative positional embeddings for all query-key pairs.

# Processing Steps
1. Computes the maximum number of relative positions needed: `2 * max(q_size, k_size) - 1`.
2. Checks if the provided `rel_pos` has the correct length (`L`). If not, resizes it using `resize_rel_pos` to match the target length.
3. Computes floating-point coordinates for the query and key positions, scaled to match each other when sizes differ.
4. Computes relative coordinates between each query and key position.
5. Adjusts the coordinates to be used as indices and casts them to integers.
6. Constructs a 3D tensor where each slice along the last dimension contains the relative positional embedding for that relative distance.
"""
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
		digits = 1,
	))

	# Calcolo le coordinate delle patch k
	k_coords = Float32.(round.(
		reshape(0:k_size-1, 1, :) * max(q_size / k_size, 1.0),
		digits = 1,
	))

	# Calcolo la differenza tra le coordinate delle patch q e k
	relative_coords =
		((q_coords .- k_coords) .+
		 ((k_size - 1) * max(q_size / k_size, 1.0))) .+ 1


	# Tronca i valori per poterli utilizzare come indici
	relative_coords = Int64.(trunc.(relative_coords))

	result = zeros(
		Float32,
		size(relative_coords, 1),
		size(relative_coords, 2),
		size(rel_pos_resized, 2),
	)


	for i in 1:size(rel_pos_resized, 2)
		result[:, :, i] = rel_pos_resized[relative_coords, i]
	end

	# Restituisce la matrice 3D dei valori degli embeddings
	return result
end


"""
########################################################
# resize_rel_pos
########################################################

    resize_rel_pos(rel_pos::AbstractArray{T, 2}, max_rel_dist::Int) where T

Interpolates a 2D matrix of relative positional embeddings to match a target maximum relative distance.

# Arguments
- `rel_pos::AbstractArray{T, 2}`: A matrix of shape (L, C), where L is the number of original relative positions
  and C is the embedding dimension.
- `max_rel_dist::Int`: The target number of relative positions after resizing.

# Returns
- `AbstractArray{Float32, 2}`: A resized matrix of shape (max_rel_dist, C) containing interpolated embeddings.

# Description
This function rescales the input relative positional encoding matrix to the desired `max_rel_dist` using linear interpolation.
"""
function resize_rel_pos(rel_pos::AbstractArray{T, 2}, max_rel_dist::Int) where T

	in_len = size(rel_pos, 1) # Taglia prima dimensione di rel_pos
	cols = size(rel_pos, 2) # Taglia seconda dimensione di rel_pos

    scale = in_len / max_rel_dist # Rapporto tra lunghezze di input e output
    x_new =  [scale * (i - 0.5) + 0.5 for i in 1:max_rel_dist]
	
    # Viene creato l'oggetto per l'interpolazione lineare
	# per simulare il comportamento di interpolazione di SAM
	# É stato sfruttato OnGrid e Flat per i valori al di fuori dei bordi
	# Non bastava una semplice interpolazione lineare
	itp = interpolate(rel_pos, BSpline(Linear()), OnGrid())
	eitp = extrapolate(itp, Flat())

	# Viene applicato lo scale per accedere via coordinate reali
	sitp = scale(eitp, 1:in_len, 1:cols)

	# Viene eseguita l’interpolazione
	res = Float32[sitp[x, j] for x in x_new, j in 1:cols]

	# Restituisce la matrice interpolata
	return res

end


"""
########################################################
# PatchEmbed
########################################################

    struct PatchEmbed

A module that embeds input images into patch embeddings using a convolutional layer.

# Fields
- `proj::Conv`: Convolutional layer that projects input patches to embedding space.
- `proj_ps::NamedTuple`: Parameters for convolution, used during forward pass.
- `proj_st::NamedTuple`: State information for convolution, used during forward pass.
"""
struct PatchEmbed
	proj::Conv # proj è un campo di tipo Conv
	proj_ps::NamedTuple
	proj_st::NamedTuple
end



"""
    PatchEmbed(; kernel_size::Tuple{Int, Int} = (16, 16),
                stride::Tuple{Int, Int} = (16, 16),
                padding::Tuple{Int, Int} = (0, 0),
                in_chans::Int = 3,
                embed_dim::Int = 768)

Constructor for `PatchEmbed`.

# Arguments
- `kernel_size::Tuple{Int, Int}`: Size of the convolutional kernel (height, width). Defaults to (16, 16).
- `stride::Tuple{Int, Int}`: Stride of the convolution. Defaults to (16, 16).
- `padding::Tuple{Int, Int}`: Padding applied to the input (height, width). Defaults to (0, 0).
- `in_chans::Int`: Number of input channels. Defaults to 3 (e.g., RGB images).
- `embed_dim::Int`: Number of output embedding dimensions. Defaults to 768.

# Returns
- An instance of `PatchEmbed` initialized with a convolutional layer configured to embed input patches.

# Description
The constructor creates a convolutional layer (`proj`) with specified kernel size, stride, and padding.
"""
function PatchEmbed(;
	kernel_size::Tuple{Int, Int} = (16, 16),
	stride::Tuple{Int, Int} = (16, 16),
	padding::Tuple{Int, Int} = (0, 0),
	in_chans::Int = 3,
	embed_dim::Int = 768,
)
	# Crea un layer di convoluzione con i parametri specificati
	# Conv viene chiamato con:
	# function Conv(w::AbstractArray{T,N}, b = true, σ = identity;
	#    stride = 1, pad = 0, dilation = 1, groups = 1) where {T,N}
	proj = Conv(
		(kernel_size...,), # (Kernel_size..,) è una tupla di interi separati
		in_chans => embed_dim;
		stride = stride,
		pad = padding,
		cross_correlation = true,
		init_weight = kaiming_uniform,
	)

	proj_ps, proj_st = Lux.setup(Random.MersenneTwister(), proj)

	# Restituisce un'istanza di PatchEmbed con il layer di convoluzione creato
	return PatchEmbed(proj, proj_ps, proj_st)
end


"""
    (self::PatchEmbed)(x::AbstractArray)

Applies the patch embedding convolution to the input tensor.

# Arguments
- `x::AbstractArray`: Input tensor with shape assumed to be (Batch, Height, Width, Channels).

# Returns
- Output tensor with patches embedded into the specified embedding dimension.
  
# Description
- The input tensor is permuted from (Batch, Height, Width, Channels) to (Channels, Batch, Height, Width) to fit the convolution layer input expectations.
- The convolution layer (`proj`) is applied.
- The output is permuted back to (Batch, Height, Width, Embedding_dim).
"""
function (self::PatchEmbed)(x::AbstractArray)
	x = permutedims(x, (3, 4, 2, 1))
	y, _ = self.proj(x, self.proj_ps, self.proj_st)
	return permutedims(y, (4, 1, 2, 3))
end




"""
########################################################
# imageEncoderViT:
########################################################

    struct ImageEncoderViT

Vision Transformer (ViT) encoder for images, including patch embedding, transformer blocks, 
positional embeddings, and a convolutional neck for feature refinement.

# Fields
- `img_size::Int`: Size of the input square image.
- `patch_embed::PatchEmbed`: Module to extract patch embeddings from the image.
- `pos_embed::Union{Nothing, Array{Float32}}`: Optional absolute positional embeddings.
- `blocks::Vector{Block}`: Sequence of transformer blocks processing the patches.
- `neck::Chain`: Multi-layer convolutional module for feature refinement.
- `neck_ps::NamedTuple`: Parameters for the `neck` module.
- `neck_st::NamedTuple`: State for the `neck` module.
"""
struct ImageEncoderViT
	img_size::Int
	patch_embed::PatchEmbed
	pos_embed::Union{Nothing, Array{Float32}}
	blocks::Vector{Block}
	neck::Chain
	neck_ps::NamedTuple
	neck_st::NamedTuple
end



"""
    ImageEncoderViT(; img_size::Int=1024,
                   patch_size::Int=16,
                   in_chans::Int=3,
                   embed_dim::Int=768,
                   depth::Int=12,
                   num_heads::Int=12,
                   mlp_ratio::Float32=4.0f0,
                   out_chans::Int=256,
                   qkv_bias::Bool=true,
                   norm_layer::Type=LayerNorm,
                   act_layer::Function=gelu_exact,
                   use_abs_pos::Bool=true,
                   use_rel_pos::Bool=false,
                   rel_pos_zero_init::Bool=true,
                   window_size::Int=0,
                   global_attn_indexes::NTuple{N, Int} where N=())

Constructor for `ImageEncoderViT`.

# Arguments
- `img_size`: Size of the square input image (default 1024).
- `patch_size`: Size of each square patch (default 16).
- `in_chans`: Number of input channels (default 3).
- `embed_dim`: Dimension of patch embeddings (default 768).
- `depth`: Number of transformer blocks (default 12).
- `num_heads`: Number of multi-head attention heads (default 12).
- `mlp_ratio`: MLP expansion ratio relative to `embed_dim` (default 4.0).
- `out_chans`: Number of output channels after convolutional neck (default 256).
- `qkv_bias`: Use bias for QKV matrices (default true).
- `norm_layer`: Type of normalization layer (default `LayerNorm`).
- `act_layer`: Activation function (default `gelu_exact`).
- `use_abs_pos`: Use absolute positional embeddings (default true).
- `use_rel_pos`: Use relative positional embeddings (default false).
- `rel_pos_zero_init`: Initialize relative positional embeddings to zero (default true).
- `window_size`: Window size for local attention (default 0).
- `global_attn_indexes`: Indices of blocks with global attention (default empty).

# Returns
- An instance of `ImageEncoderViT` ready for the forward pass.
"""
function ImageEncoderViT(;
	img_size::Int = 1024,
	patch_size::Int = 16,
	in_chans::Int = 3,
	embed_dim::Int = 768,
	depth::Int = 12,
	num_heads::Int = 12,
	mlp_ratio::Float32 = 4.0f0,
	out_chans::Int = 256,
	qkv_bias::Bool = true,
	norm_layer::Type = LayerNorm,
	act_layer::Function = gelu_exact,
	use_abs_pos::Bool = true,
	use_rel_pos::Bool = false,
	rel_pos_zero_init::Bool = true,
	window_size::Int = 0,
	global_attn_indexes::NTuple{N, Int} where N = (),
)

	patch_embed = PatchEmbed(
		kernel_size = (patch_size, patch_size),
		stride = (patch_size, patch_size),
		in_chans = in_chans,
		embed_dim = embed_dim,
	)

	pos_embed = nothing

	if use_abs_pos
		pos_embed = zeros(
			Float32,
			1,
			img_size ÷ patch_size,
			img_size ÷ patch_size,
			embed_dim,
		)
	end

	blocks = Vector{Block}(undef, depth)

	for i in 1:depth
		blocks[i] = Block(
			dim = embed_dim,
			num_heads = num_heads,
			mlp_ratio = mlp_ratio,
			qkv_bias = qkv_bias,
			norm_layer = norm_layer,
			act_layer = act_layer,
			use_rel_pos = use_rel_pos,
			rel_pos_zero_init = rel_pos_zero_init,
			window_size = i ∉ global_attn_indexes ? window_size : 0,
			input_size = (img_size ÷ patch_size, img_size ÷ patch_size),
		)
	end

	neck = Chain(
		Conv((1, 1), 
            embed_dim => out_chans, 
            use_bias = false, 
            cross_correlation = true
            ),
		LayerNorm2d(out_chans), 
        Conv(
            (3, 3), 
            out_chans => out_chans, 
            pad = 1, 
            use_bias = false,
			cross_correlation = true
            ),
		LayerNorm2d(out_chans),
	)

	neck_ps, neck_st = Lux.setup(Random.MersenneTwister(), neck)

	return ImageEncoderViT(
		img_size,
		patch_embed,
		pos_embed,
		blocks,
		neck,
		neck_ps,
		neck_st,
	)
end


"""
    (self::ImageEncoderViT)(x::AbstractArray)

Forward pass of the ViT image encoder.

# Arguments
- `x::AbstractArray`: Input image tensor with dimensions (Batch, Height, Width, Channels).

# Returns
- Feature tensor after processing through transformer blocks and convolutional neck,
  with dimensions (Batch, Channels, Height, Width).

# Description
- Applies patch embedding to the input.
- Adds absolute positional embeddings if present.
- Passes the result through each transformer block.
- Applies the convolutional neck module with necessary dimension permutations.
"""
function (self::ImageEncoderViT)(x::AbstractArray)

	x = self.patch_embed(x)

	if !isnothing(self.pos_embed)
		x = x .+ self.pos_embed
	end

	for i in 1:length(self.blocks)
		println(i)
		x = self.blocks[i](x)
	end

	x = permutedims(x, (2, 3, 4, 1))

	x, _ = Lux.apply(self.neck, x, self.neck_ps, self.neck_st)

	return permutedims(x, (4, 3, 1, 2))
end

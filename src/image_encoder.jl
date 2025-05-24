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

########################################################
# Block:
########################################################

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


########################################################
# Attention: 
########################################################

# Definizione struttura attention
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


########################################################
# resize_rel_pos: 
# Restituisce l' interpolazione ottuta per scalare la matrice
########################################################

# Viene definita la funzione resize_rel_pos
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



########################################################
# PatchEmbed: definito per implementare l'embedding delle patch dell'immagine
# con embedding si intende la rappresentazione delle patch come vettori numerici
# Serve per applicare una convoluzione 2D alle patch dell'immagine
########################################################

# Viene definito un tipo strutturato PatchEmbed
# Rappresenta un layer di embedding per le patch dell'immagine
struct PatchEmbed
	proj::Conv # proj è un campo di tipo Conv
	proj_ps::NamedTuple
	proj_st::NamedTuple
end

# Costruttore per PatchEmbed con parametri di default
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

# Viene definito il forward pass per PatchEmbed
function (self::PatchEmbed)(x::AbstractArray)
	x = permutedims(x, (3, 4, 2, 1))
	y, _ = self.proj(x, self.proj_ps, self.proj_st)
	return permutedims(y, (4, 1, 2, 3))
end



########################################################
# imageEncoderViT:
########################################################

struct ImageEncoderViT
	img_size::Int
	patch_embed::PatchEmbed
	pos_embed::Union{Nothing, Array{Float32}}
	blocks::Vector{Block}
	neck::Chain
	neck_ps::NamedTuple
	neck_st::NamedTuple
end


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


function (self::ImageEncoderViT)(x::AbstractArray)

	x = self.patch_embed(x)

	if !isnothing(self.pos_embed)
		x = x .+ self.pos_embed
	end

	for i in 1:length(self.blocks)
		x = self.blocks[i](x)
	end

	x = permutedims(x, (2, 3, 4, 1))

	x, _ = Lux.apply(self.neck, x, self.neck_ps, self.neck_st)

	return permutedims(x, (4, 3, 1, 2))
end

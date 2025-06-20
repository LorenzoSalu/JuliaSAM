using SpecialFunctions
using LoopVectorization
using ImageTransformations

# Funzione per la permutazione degli array in ordine inverso
function reverse_pos(a::AbstractArray)
	return permutedims(a, reverse(1:ndims(a)))
end

# Funzione per il reshape python-like
function sam_reshape(a, dims)
	b = PermutedDimsArray(a, reverse(1:ndims(a)))
	a = reshape(b, reverse(dims))
	a = permutedims(a, reverse(1:ndims(a)))
	return a
end

# Funzione per il calcolo gelu non approssimato
function gelu_exact(x)
	return 0.5f0 * x * (1.0f0 + erf(x / sqrt(2.0f0)))
end


function safe_imresize(arr, new_size; method = Linear())
	# Trova dimensioni non-singleton
	non_singleton_dims = findall(size(arr) .> 1)

	if length(non_singleton_dims) == length(new_size)
		# Estrai solo le dimensioni spaziali
		spatial_data =
			dropdims(arr, dims = tuple(findall(size(arr) .== 1)...))

		# Ridimensiona
		resized = imresize(spatial_data, new_size, method = method)

		# Ricostruisci con dimensioni originali
		original_shape = collect(size(arr))
		original_shape[non_singleton_dims] .= new_size
		return reshape(resized, original_shape...)
	else
		error("Mismatch between non-singleton dimensions and target size")
	end
end



function to_pil_image(image)
	"""
	Converte:
	- tensor di shape (C, H, W) 
	- tensor di shape (H, W, C)
	- In PIL Image format (H, W, C) con Int

	Gestisce automaticamente:
	- Conversione range [0,1] -> [0,255] 
	- Conversione tipo Float -> Int
	- Riordinamento dimensioni CHW -> HWC
	"""

	is_chw_format = false
	if ndims(image) == 3
		c, h, w = size(image)

		if c <= 4 && h > c && w > c
			is_chw_format = true
		end
	end


	# Controlla il range dei valori
	min_val, max_val = extrema(image)

	if 0 <= min_val && max_val <= 1
		# Range [0,1] -> [0,255]
		image = image .* 255
	elseif -1 <= min_val && max_val <= 1
		# Range [-1,1] -> [0,255]
		image = (image .+ 1) .* 127.5
	end

	# Converti a Int come fa PIL
	image = trunc.(Int, clamp.(image, 0, 255))


	# 3. Converti da CHW a HWC se necessario
	if is_chw_format
		image = permutedims(image, (2, 3, 1))  # CHW -> HWC
	end

	# 4. Gestisci immagini grayscale
	if ndims(image) == 2
		# Mantieni 2D per grayscale (PIL accetta sia 2D che 3D con 1 canale)
		image = image
	elseif ndims(image) == 3 && size(image, 3) == 1
		# Rimuovi dimensione singleton per grayscale
		image = dropdims(image, dims = 3)
	end

	return trunc.(Int, image)
end


###########################################################
# Assegnazione pesi e bias dal checkpoint
###########################################################

function assign_weight!(model, path_str::String, value)
	parts = split(path_str, '.')
	obj = model

	for i ∈ 1:length(parts)-1
		part = parts[i]

		# Gestione dell'indice (es: layers_ps[2])
		if occursin('[', part)
			base, idx_str = match(r"(\w+)\[(\d+)\]", part).captures
			idx = parse(Int, idx_str)
			obj = getproperty(obj, Symbol(base))[idx]
		else
			obj = getproperty(obj, Symbol(part))
		end
	end

	# Ultimo campo: potrebbe essere .weight, .bias, ecc.
	last_part = parts[end]
	if occursin('[', last_part)
		base, idx_str = match(r"(\w+)\[(\d+)\]", last_part).captures
		idx = parse(Int, idx_str)
		getproperty(obj, Symbol(base))[idx] .= value
	else
		getproperty(obj, Symbol(last_part)) .= value
	end
end



function load_model_weights!(
	model::Any,
	state_dict::Dict{String, Array{Float32}},
)

	for (k, v) in state_dict
		try
			assign_weight!(model, k, v)
		catch e
			@warn "Errore assegnando $k" exception = e
		end
	end
end


##################################################
# Funzione per facilitare i test
##################################################

function is_correct(string::String, x::AbstractArray, y::AbstractArray)
	println(
		string,
		all(isapprox.(
			x,
			y;
			atol = 0.4,
			rtol = 0)),
		"	 Differenza massima: ",
		maximum(abs.(x .- y)),
	)
end






##################

function torchvision_resize_exact(img, img_size;
	interpolation = :bilinear, align_corners = false, antialias = false)
	"""
	Versione corretta per matching ESATTO con torchvision

	CORREZIONI CRITICHE:
	1. Float32 invece di Float64 (torchvision usa float32)
	2. Coordinate mapping esatto secondo il codice C++ di PyTorch
	3. Arrotondamento finale identico
	4. Gestione corretta dei tipi di dato
	"""

	# Gestione formato input
	original_shape = size(img)
	ndims_orig = ndims(img)

	if ndims_orig == 2
		h_orig, w_orig = original_shape
		n_channels = 1
		batch_size = 1
		img_work = reshape(Float32.(img), h_orig, w_orig, 1, 1)
	elseif ndims_orig == 3
		h_orig, w_orig, n_channels = original_shape
		batch_size = 1
		img_work = reshape(Float32.(img), h_orig, w_orig, n_channels, 1)
	elseif ndims_orig == 4
		h_orig, w_orig, n_channels, batch_size = original_shape
		img_work = Float32.(img)
	else
		error("Input deve essere 2D, 3D o 4D")
	end

	# Calcola dimensioni target
	if isa(img_size, Int)
		if h_orig < w_orig
			h_new = img_size
			w_new = round(Int, w_orig * img_size / h_orig)
		else
			w_new = img_size
			h_new = round(Int, h_orig * img_size / w_orig)
		end
	else
		h_new, w_new = img_size
	end

	# Alloca output in Float32
	output = zeros(Float32, h_new, w_new, n_channels, batch_size)

	# CORREZIONE: Scalari precisi come torchvision
	scale_y = Float32(h_orig) / Float32(h_new)
	scale_x = Float32(w_orig) / Float32(w_new)

	# Main resize loop
	for batch_idx in 1:batch_size
		for channel_idx in 1:n_channels
			for i in 1:h_new
				for j in 1:w_new
					# COORDINATE MAPPING ESATTO
					if align_corners
						if h_new == 1
							src_y = Float32(h_orig + 1) / 2.0f0
						else
							src_y = Float32(i - 1) * Float32(h_orig - 1) / Float32(h_new - 1) + 1.0f0
						end
						if w_new == 1
							src_x = Float32(w_orig + 1) / 2.0f0
						else
							src_x = Float32(j - 1) * Float32(w_orig - 1) / Float32(w_new - 1) + 1.0f0
						end
					else
						# FORMULA ESATTA da PyTorch C++ (align_corners=False)
						# coordinate = (output_index + 0.5) * scale - 0.5
						src_y_0indexed = 
                            (Float32(i - 1) + 0.5f0) * scale_y - 0.5f0
						src_x_0indexed = 
                            (Float32(j - 1) + 0.5f0) * scale_x - 0.5f0

						# Converti a 1-indexed per Julia, ma mantieni Float32
						src_y = src_y_0indexed + 1.0f0
						src_x = src_x_0indexed + 1.0f0

						# NESSUN CLAMPING qui - lo fa l'interpolazione
					end

					# Applica interpolazione
					if interpolation == :bilinear
						output[i, j, channel_idx, batch_idx] =
							bilinear_interpolation_exact(
								img_work, src_y, src_x,
								channel_idx, batch_idx, h_orig, w_orig,
							)
					else
						error("Solo bilinear supportato in questa 
							versione exact")
					end
				end
			end
		end
	end

	# CONVERSIONE FINALE ESATTA come PyTorch
	# PyTorch usa "round to nearest even" (banker's rounding) internamente
	# ma poi fa una conversione specifica per UInt8

	# CRITICO: PyTorch potrebbe usare truncation invece di round
	# Proviamo diverse strategie
	# test_rounding_strategies(out)
	output_uint8 = trunc.(UInt8, round.(clamp.(output, 0.0f0, 255.0f0)))



	# Reshape output per matchare input
	if ndims_orig == 2
		return reshape(output_uint8, h_new, w_new)
	elseif ndims_orig == 3
		return reshape(output_uint8, h_new, w_new, n_channels)
	else
		return output_uint8
	end
end

function bilinear_interpolation_exact(img, src_y, src_x,
	channel_idx, batch_idx, h_orig, w_orig)
	"""
	Interpolazione bilineare ESATTA come PyTorch C++
	CRITICO: gestione identica dei boundary conditions
	"""

	# Converti a coordinate 0-indexed per calcoli
	src_y_0 = src_y - 1.0f0
	src_x_0 = src_x - 1.0f0

	# Floor per coordinate intere (0-indexed)
	y_low_0 = floor(Int32, src_y_0)  # Int32 come PyTorch
	x_low_0 = floor(Int32, src_x_0)
	y_high_0 = y_low_0 + 1
	x_high_0 = x_low_0 + 1

	# BOUNDARY HANDLING IDENTICO A PYTORCH
	# PyTorch clamps DOPO aver calcolato high coordinates
	y_low_0 = max(0, min(h_orig - 1, y_low_0))
	y_high_0 = max(0, min(h_orig - 1, y_high_0))
	x_low_0 = max(0, min(w_orig - 1, x_low_0))
	x_high_0 = max(0, min(w_orig - 1, x_high_0))

	# Converti a 1-indexed per Julia
	y_low = y_low_0 + 1
	y_high = y_high_0 + 1
	x_low = x_low_0 + 1
	x_high = x_high_0 + 1

	# Calcola pesi ESATTO
	# CRITICO: usa le coordinate originali non clampate per i pesi
	ly = src_y_0 - Float32(floor(Int32, src_y_0))
	lx = src_x_0 - Float32(floor(Int32, src_x_0))
	hy = 1.0f0 - ly
	hx = 1.0f0 - lx

	# Leggi valori pixel
	v1 = img[y_low, x_low, channel_idx, batch_idx]
	v2 = img[y_low, x_high, channel_idx, batch_idx]
	v3 = img[y_high, x_low, channel_idx, batch_idx]
	v4 = img[y_high, x_high, channel_idx, batch_idx]

	# Interpolazione bilineare - formula PyTorch
	w1 = hy * hx
	w2 = hy * lx
	w3 = ly * hx
	w4 = ly * lx

	return v1 * w1 + v2 * w2 + v3 * w3 + v4 * w4
end


# FUNZIONI PER TEST DIVERSE STRATEGIE DI ARROTONDAMENTO
function test_rounding_strategies(float_output, expected_result)
	"""
	Testa diverse strategie di arrotondamento per trovare quella corretta
	"""
	println("🧪 TEST STRATEGIE ARROTONDAMENTO")
	println("="^50)

	# Clamp comune
	clamped = clamp.(float_output, 0.0f0, 255.0f0)

	strategies = [
		("round", x -> round(UInt8, x)),
		("floor+0.5", x -> UInt8(floor(x + 0.5f0))),
		("trunc+0.5", x -> UInt8(trunc(x + 0.5f0))),
		("ceil-0.5", x -> UInt8(ceil(x - 0.5f0))),
		("simple_cast", x -> UInt8(x)),
		("round_banker", x -> UInt8(round(x, RoundNearestTiesAway))),
		("round_half_up", x -> UInt8(floor(x + 0.5f0))),
		("pytorch_style", x -> UInt8(round(x))),
	]

	for (name, func) in strategies
		try
			result = func.(clamped)
			diff = Int.(result) .- Int.(expected_result)
			identical = sum(diff .== 0)
			total = length(diff)
			accuracy = round(identical / total * 100, digits = 2)

			println("$name: $accuracy% identici ($identical/$total)")

			if accuracy == 100.0
				println("🎯 STRATEGIA PERFETTA TROVATA: $name")
				return name, func
			end
		catch e
			println("$name: ERRORE - $e")
		end
	end

	return nothing, nothing
end

function torchvision_resize_with_strategy(img, img_size;
	interpolation = :bilinear, align_corners = false)
	"""
	Versione che permette di testare diverse strategie di arrotondamento
	"""
	# Gestione formato input
	original_shape = size(img)
	ndims_orig = ndims(img)

	if ndims_orig == 2
		h_orig, w_orig = original_shape
		n_channels = 1
		batch_size = 1
		img_work = reshape(Float32.(img), h_orig, w_orig, 1, 1)
	elseif ndims_orig == 3
		h_orig, w_orig, n_channels = original_shape
		batch_size = 1
		img_work = reshape(Float32.(img), h_orig, w_orig, n_channels, 1)
	elseif ndims_orig == 4
		h_orig, w_orig, n_channels, batch_size = original_shape
		img_work = Float32.(img)
	else
		error("Input deve essere 2D, 3D o 4D")
	end

	# Calcola dimensioni target
	if isa(img_size, Int)
		if h_orig < w_orig
			h_new = img_size
			w_new = round(Int, w_orig * img_size / h_orig)
		else
			w_new = img_size
			h_new = round(Int, h_orig * img_size / w_orig)
		end
	else
		h_new, w_new = img_size
	end

	# Alloca output in Float32
	output = zeros(Float32, h_new, w_new, n_channels, batch_size)

	# CORREZIONE: Scalari precisi come torchvision
	scale_y = Float32(h_orig) / Float32(h_new)
	scale_x = Float32(w_orig) / Float32(w_new)

	# Main resize loop
	for batch_idx in 1:batch_size
		for channel_idx in 1:n_channels
			for i in 1:h_new
				for j in 1:w_new
					# COORDINATE MAPPING ESATTO
					if align_corners
						if h_new == 1
							src_y = Float32(h_orig + 1) / 2.0f0
						else
							src_y = Float32(i - 1) * Float32(h_orig - 1) / Float32(h_new - 1) + 1.0f0
						end
						if w_new == 1
							src_x = Float32(w_orig + 1) / 2.0f0
						else
							src_x = Float32(j - 1) * Float32(w_orig - 1) / Float32(w_new - 1) + 1.0f0
						end
					else
						# FORMULA ESATTA da PyTorch C++ (align_corners=False)
						# coordinate = (output_index + 0.5) * scale - 0.5
						src_y_0indexed = (Float32(i - 1) + 0.5f0) * scale_y - 0.5f0
						src_x_0indexed = (Float32(j - 1) + 0.5f0) * scale_x - 0.5f0

						# Converti a 1-indexed per Julia, ma mantieni Float32
						src_y = src_y_0indexed + 1.0f0
						src_x = src_x_0indexed + 1.0f0

						# NESSUN CLAMPING qui - lo fa l'interpolazione
					end

					# Applica interpolazione
					if interpolation == :bilinear
						output[i, j, channel_idx, batch_idx] = bilinear_interpolation_exact(
							img_work, src_y, src_x, channel_idx, batch_idx, h_orig, w_orig,
						)
					else
						error("Solo bilinear supportato in questa versione exact")
					end
				end
			end
		end
	end

	output_clamped = clamp.(output, 0.0f0, 255.0f0)

	test_rounding_strategies(output_clamped, expected_input_image)

	# Reshape output per matchare input
	if ndims_orig == 2
		return reshape(output_uint8, h_new, w_new)
	elseif ndims_orig == 3
		return reshape(output_uint8, h_new, w_new, n_channels)
	else
		return output_uint8
	end
end

# FUNZIONE PER CONFRONTO DIRETTO
function compare_with_expected(your_result, expected_result)
	"""
	Confronto pixel per pixel con analisi dettagliata
	"""
	diff = Int.(your_result) .- Int.(expected_result)

	# Trova il primo pixel diverso
	diff_indices = findall(x -> x != 0, diff)

	if length(diff_indices) > 0
		println("Primi 10 pixel diversi:")
		for i in 1:min(10, length(diff_indices))
			idx = diff_indices[i]
			your_val = your_result[idx]
			exp_val = expected_result[idx]
			diff_val = diff[idx]
			println("Pos $idx: Your=$your_val, 
				Expected=$exp_val, Diff=$diff_val")
		end

		# Analizza distribution delle differenze
		unique_diffs = unique(diff)
		println("\nDistribuzione differenze:")
		for d in sort(unique_diffs)
			count = sum(diff .== d)
			percentage = round(count / length(diff) * 100, digits = 2)
			println("$d: $count pixel ($percentage%)")
		end

		# Controlla se sono tutti ±1 (problema di arrotondamento)
		if all(abs.(unique_diffs) .<= 1)
			println("\n⚠️  SOLO DIFFERENZE ±1: Problema di arrotondamento!")
			println("Suggerimento: usa test_rounding_strategies() per trovare la strategia corretta")
		end
	else
		println("✅ PERFETTO! Nessuna differenza!")
	end

	return diff
end

# FUNZIONE PER ANALIZZARE I VALORI FLOAT PRIMA DELLA CONVERSIONE
function analyze_float_values_before_conversion(float_output,
	expected_result, sample_indices = nothing)
	"""
	Analizza i valori float prima della conversione per capire il pattern di arrotondamento
	"""
	if sample_indices === nothing
		# Prendi alcuni indici dove ci sono differenze note
		sample_indices = [
			CartesianIndex(155, 1, 1),
			CartesianIndex(203, 1, 1),
			CartesianIndex(218, 1, 1),
			CartesianIndex(5, 2, 1),
			CartesianIndex(22, 2, 1),
		]
	end

	println("🔍 ANALISI VALORI FLOAT PRE-CONVERSIONE")
	println("="^60)

	for idx in sample_indices
		if checkbounds(Bool, float_output, idx) && checkbounds(Bool, expected_result, idx)
			float_val = float_output[idx]
			expected_val = expected_result[idx]

			# Test diverse conversioni per questo valore
			strategies = [
				round(UInt8, float_val),
				UInt8(floor(float_val + 0.5f0)),
				UInt8(trunc(float_val + 0.5f0)),
				UInt8(floor(float_val)),
				UInt8(ceil(float_val)),
			]

			println("Pos $idx:")
			println("  Float value: $float_val")
			println("  Expected: $expected_val")
			println("  round(): $(strategies[1]) $(strategies[1] == expected_val ? "✅" : "❌")")
			println("  floor+0.5: $(strategies[2]) $(strategies[2] == expected_val ? "✅" : "❌")")
			println("  trunc+0.5: $(strategies[3]) $(strategies[3] == expected_val ? "✅" : "❌")")
			println("  floor(): $(strategies[4]) $(strategies[4] == expected_val ? "✅" : "❌")")
			println("  ceil(): $(strategies[5]) $(strategies[5] == expected_val ? "✅" : "❌")")
			println()
		end
	end
end

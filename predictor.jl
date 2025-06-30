using Lux
using CUDA
using Interpolations
using TensorOperations
using Einsum
using NPZ
using NNlib
using ImageTransformations
using ImageComponentAnalysis
using ChainRulesCore
using OpenCV, FileIO
using CairoMakie
using Colors, ColorTypes
using FixedPointNumbers

using PyCall

using Profile, ProfileView

include("./global_functions.jl")
include("./utils/amg.jl")
include("./utils/transforms.jl")
include("./src/Sam.jl")
include("./build_sam.jl")



function overlay_mask(ax, mask::AbstractMatrix{Bool}; color = RGBAf(30 / 255, 144 / 255, 1, 0.6))
	h, w = size(mask)
	mask_img = Array{RGBAf}(undef, w, h)
	for y in 1:h, x in 1:w
		mask_img[x, y] = mask[y, x] ? color : RGBAf(0, 0, 0, 0)
	end
	image!(ax, mask_img; transparency = true)
end


function overlay_points(ax, coords::AbstractArray, labels::Vector{Bool};
	marker_size = 20, edgecolor = :white)
	# separa positivi (true) e negativi (false)
    pos = coords[labels .== true, :]
    neg = coords[labels .== false, :]

	scatter!(ax, pos; marker = :star5, color = :green, markersize = marker_size, strokecolor = edgecolor)
	scatter!(ax, neg; marker = :star5, color = :red, markersize = marker_size, strokecolor = edgecolor)
end


function overlay_box(ax, box::NTuple{4,Float64}; color=:green, linewidth=2)
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    rect = Rectangle(x0, y0, w, h)
    poly!(ax, rect; strokecolor=color, fillcolor=RGBAf(0,0,0,0), linewidth=linewidth)
end

########################################################
########################################################
# SamPredictor
########################################################
########################################################


mutable struct SamPredictor
	model::Sam
	transform::ResizeLongestSize
	original_size::Union{Nothing, Tuple{Int, Int}}
	features::Union{Nothing, AbstractArray}
	is_image_set::Bool
	input_size::Union{Nothing, Tuple{Int, Int}}
	orig_h::Union{Nothing, <:Any}
	orig_w::Union{Nothing, <:Any}
	input_h::Union{Nothing, <:Any}
	input_w::Union{Nothing, <:Any}
	device::Union{Nothing, String}
end

function SamPredictor(
	sam_model::Sam,
)
	"""
	Uses SAM to calculate the image embedding for an image, and then
	allow repeated, efficient mask prediction given prompts.

	Arguments:
		sam_model (Sam): The model to use for mask prediction.
	"""

	transform = ResizeLongestSize(sam_model.image_encoder.img_size)

	predictor = SamPredictor(
		sam_model,
		transform,
		nothing,
		nothing,
		false,
		nothing,
		nothing,
		nothing,
		nothing,
		nothing,
		sam_model.device,
	)

	reset_image(predictor)

	return predictor
end



function set_image(
	self::SamPredictor,
	image::AbstractArray,
	image_format::String = "RGB",
)
	"""
	Calculates the image embeddings for the provided image, allowing
	masks to be predicted with the 'predict' method.

	Arguments:
		image (np.ndarray): The image for calculating masks. Expects an
		image in HWC uint8 format, with pixel values in [0, 255].
		image_format (str): The color format of the image, in ['RGB', 'BGR'].
	"""

	@assert image_format in ["RGB", "BGR"]
	"image_format must be in ['RGB', 'BGR'], is $image_format."

	if image_format != self.model.image_format
		image = image[:, :, end:-1:1]
	end

	#is_correct("image corretto? ", image, expected_image)

	input_image = apply_image(self.transform, image)

	#compare_with_expected(input_image, expected_input_image)

	#is_correct("input_image corretto? ",input_image, expected_input_image)
	#println(count(input_image .!= expected_input_image), " / ", length(input_image))


	if self.device == "gpu"
		input_image_array = CuArray(input_image)
	else
		input_image_array = input_image
	end

	input_image_array = permutedims(input_image_array, (3, 1, 2))
	input_image_array =
		reshape(input_image_array, (1, size(input_image_array)...))

	#is_correct("input_image_array corretto? ",input_image_array, expected_input_image_array)

	set_array_image(self, input_image_array, size(image)[1:2])
end




function set_array_image(
	self::SamPredictor,
	transformed_image::AbstractArray,
	original_image_size::Tuple{Int, Int},
)
	"""
	Calculates the image embeddings for the provided image, allowing
	masks to be predicted with the 'predict' method. Expects the input
	image to be already transformed to the format expected by the model.

	Arguments:
		transformed_image (torch.Tensor): The input image, with shape
		1x3xHxW, which has been transformed with ResizeLongestSide.
		original_image_size (tuple(int, int)): The size of the image
		before transformation, in (H, W) format.
	"""

	return ChainRulesCore.ignore_derivatives() do

		@assert ndims(transformed_image) == 4
		"set_array_image input must be with 4 dimensions."
		@assert size(transformed_image, 2) == 3
		"set_array_image input must have 3 channels in the second dimension"
		@assert maximum(size(transformed_image)[3:4]) ==
				self.model.image_encoder.img_size
		"set_array_image input must have long side equal to 
			$(self.model.image_encoder.img_size)."

		reset_image(self)

		self.original_size = original_image_size
		self.input_size = Tuple(size(transformed_image)[end-1:end])

		input_image = preprocess(self.model, transformed_image)

		#is_correct("input_image_preprocess corretto? ",input_image, expected_input_image_preprocess)

		println("Entra image_encoder")
		self.features = self.model.image_encoder(Float32.(input_image))
		println("Fine image_encoder")

		self.is_image_set = true
	end
end

function predict(
	self::SamPredictor;
	point_coords::Union{Nothing, AbstractArray} = nothing,
	point_labels::Union{Nothing, AbstractArray} = nothing,
	box::Union{Nothing, AbstractArray} = nothing,
	mask_input::Union{Nothing, AbstractArray} = nothing,
	multimask_output::Bool = true,
	return_logits::Bool = false,
)::Tuple{AbstractArray, AbstractArray, AbstractArray}
	"""
	Predict masks for the given input prompts, using the currently set image.

	Arguments:

	point_coords (AbstractArray or None): A Nx2 array of point prompts to the
	model. Each point is in (X,Y) in pixels.
	point_labels (AbstractArray or None): A length N array of labels for the
	point prompts. 1 indicates a foreground point and 0 indicates a
	background point.
	box (AbstractArray or None): A length 4 array given a box prompt to the
	model, in XYXY format.
	mask_input (AbstractArray): A low resolution mask input to the model,
	coming from a previous prediction iteration. Has form 1xHxW, where
	for SAM, H=W=256.
	multimask_output (bool): If true, the model will return three masks.
	For ambiguous input prompts (such as a single click), this will often
	produce better masks than a single prediction. If only a single
	mask is needed, the model's predicted quality score can be used
	to select the best mask. For non-ambiguous prompts, such as multiple
	input prompts, multimask_output=False can give better results.
	return_logits (bool): If true, returns un-thresholded masks logits
	instead of a binary mask.

	Returns:

	(AbstractArray): The output masks in CxHxW format, where C is the
	number of masks, and (H, W) is the original image size.
	(AbstractArray): An array of length C containing the model's
	predictions for the quality of each mask.
	(AbstractArray): An array of shape CxHxW, where C is the number
	of masks and H=W=256. These low resolution logits can be passed to
	a subsequent iteration as mask input.
	"""

	if !self.is_image_set
		error(
			"An image must be set with .set_image(...) before mask prediction.")
	end

	coords_array, labels_array, box_array, mask_input_array =
		nothing, nothing, nothing, nothing

	if point_coords !== nothing
		@assert point_labels !== nothing
		"point_labels must be supplied if point_coords is supplied."
		point_coords =
			apply_coords(self.transform, point_coords, self.original_size)

		#is_correct("apply_coords corretto? ",point_coords, expected_apply_coords)

		if self.device == "gpu"
			coords_array = CuArray(Float32.(point_coords))
			labels_array = CuArray(trunc.(Int, point_labels))
		else
			coords_array = Float32.(point_coords)
			labels_array = trunc.(Int, point_labels)
		end

		coords_array = reshape(coords_array, (1, size(coords_array)...))
		labels_array = reshape(labels_array, (1, size(labels_array)...))
	end

	if box !== nothing
		box = apply_boxes(self.transform, box, self.original_size)
		if self.device == "gpu"
			box_array = CuArray(Float32.(box))
		else
			box_array = Float32.(box)
		end
		box_array = reshape(box_array, (1, size(box_array)...))
	end

	if mask_input !== nothing
		if self.device == "gpu"
			mask_input_array = CuArray(Float32.(mask_input))
		else
			mask_input_array = Float32.(mask_input)
		end

		mask_input_array =
			reshape(mask_input_array, (1, size(mask_input_array)...))
	end

	#is_correct("coords_array corretto? ", coords_array, expected_coords_array)
	#is_correct("labels_array corretto? ", labels_array, expected_labels_array)

	masks, iou_predictions, low_res_masks = predict_array(
		self,
		point_coords = coords_array,
		point_labels = labels_array,
		boxes = box_array,
		mask_input = mask_input_array,
		multimask_output = multimask_output,
		return_logits = return_logits,
	)

    println(size(masks))
    
	masks_np = [masks[1]]
	iou_predictions_np = [iou_predictions[1]]
	low_res_masks_np = [low_res_masks[1]]

	return masks_np, iou_predictions_np, low_res_masks_np
end


function predict_array(
	self::SamPredictor;
	point_coords::Union{Nothing, AbstractArray} = nothing,
	point_labels::Union{Nothing, AbstractArray} = nothing,
	boxes::Union{Nothing, AbstractArray} = nothing,
	mask_input::Union{Nothing, AbstractArray} = nothing,
	multimask_output::Bool = true,
	return_logits::Bool = false,
)::Tuple{AbstractArray, AbstractArray, AbstractArray}
	"""
	Predict masks for the given input prompts, using the currently set image.
	Input prompts are batched tensors and are expected to already be
	transformed to the input frame using ResizeLongestSide.

	Arguments:
	point_coords (AbstractArray or None): A BxNx2 array of point prompts to the
		model. Each point is in (X,Y) in pixels.
	point_labels (AbstractArray or None): A BxN array of labels for the
		point prompts. 1 indicates a foreground point and 0 indicates a
		background point.
	boxes (AbstractArray or None): A Bx4 array given a box prompt to the
		model, in XYXY format.
	mask_input (AbstractArray): A low resolution mask input to the model, 
		coming from a previous prediction iteration. Has form Bx1xHxW, where
		for SAM, H=W=256. Masks returned by a previous iteration of the
		predict method do not need further transformation.
	multimask_output (bool): If true, the model will return three masks.
		For ambiguous input prompts (such as a single click), this will often
		produce better masks than a single prediction. If only a single
		mask is needed, the model's predicted quality score can be used
		to select the best mask. For non-ambiguous prompts, such as multiple
		input prompts, multimask_output=False can give better results.
	return_logits (bool): If true, returns un-thresholded masks logits
		instead of a binary mask.

	Returns:
	(AbstractArray): The output masks in BxCxHxW format, where C is the
		number of masks, and (H, W) is the original image size.
	(AbstractArray): An array of shape BxC containing the model's
		predictions for the quality of each mask.
	(AbstractArray): An array of shape BxCxHxW, where C is the number
		of masks and H=W=256. These low res logits can be passed to
		a subsequent iteration as mask input.
	"""
	return ChainRulesCore.ignore_derivatives() do
		if !self.is_image_set
			error(
				"An image must be set with .set_image(...) 
				before mask prediction.")
		end

		if point_coords !== nothing
			points = (point_coords, point_labels)
		else
			points = nothing
		end

		# Embed prompts
		sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
			points = points,
			boxes = boxes,
			masks = mask_input,
		)
        #=
		is_correct("sparse_embeddings corretto? ",
			sparse_embeddings, expected_predict_sparse_embeddings)
		is_correct("dense_embeddings corretto? ",
			dense_embeddings, expected_predict_dense_embeddings)

		is_correct("self.features corretto? ",
			self.features, expected_predict_self_features)
        =#
		image_pe = get_dense_pe(self.model.prompt_encoder)

		#is_correct("get_dense_pe() corretto? ", image_pe, expected_predict_get_dense_pe)

		# Predict masks
		low_res_masks, iou_predictions = self.model.mask_decoder(
			image_embeddings = self.features,
			image_pe = image_pe,
			sparse_prompt_embeddings = sparse_embeddings,
			dense_prompt_embeddings = dense_embeddings,
			multimask_output = multimask_output,
		)

        #=
		is_correct("low_res_masks corretto? ",
			low_res_masks, expected_predict_low_res_masks)
		is_correct("iou_predictions corretto? ",
			iou_predictions, expected_predict_iou_predictions)
        =#
		# Upscale the masks to the original image resolution
		masks = postprocess_masks(
			self.model,
			masks = low_res_masks,
			input_size = self.input_size,
			original_size = self.original_size,
		)
        #=
		is_correct("masks corretto? ",
			masks, expected_predict_maskss)
        =#
		if !return_logits
			masks = masks .> self.model.mask_threshold
		end

		return masks, iou_predictions, low_res_masks
	end
end



function get_image_embedding(
	self::SamPredictor,
)
	"""
	Returns the image embeddings for the currently set image, with
	shape 1xCxHxW, where C is the embedding dimension and (H,W) are
	the embedding spatial dimension of SAM (typically C=256, H=W=64).
	"""

	if !self.is_image_set
		error("An image must be set with .set_image(...)
			to generate an embedding.")
	end

	@assert self.features !== nothing
	"Features must exist if an image has been set."

	return self.features
end

function reset_image(self::SamPredictor)
	"""Resets the currently set image."""
	self.is_image_set = false
	self.features = nothing
	self.orig_h = nothing
	self.orig_w = nothing
	self.input_h = nothing
	self.input_w = nothing
end

########################################################
# Test sulle funzioni:
########################################################

checkpoint = npzread("model_checkpoint.npz")
img_cv = 
    OpenCV.imread("./predictorTest/testFiles/truck.jpg", OpenCV.IMREAD_COLOR)
img = OpenCV.cvtColor(img_cv, OpenCV.COLOR_BGR2RGB)
img_array = Array(img)

sam = build_sam(checkpoint = checkpoint)
predictor = SamPredictor(sam)
set_image(predictor, img_array)



pts = [[500,375]]
pts = reduce(hcat, pts)'
lbl = [true]


masks, scores, logits = predict(
    predictor,
    point_coords = pts,
    point_labels = lbl,
    multimask_output = true,
)

function convert_cv_rgb(img_cv::Array{UInt8,3})
    h, w, _ = size(img_cv)
    img_rgb = Matrix{RGB{N0f8}}(undef, h, w)
    for y in 1:h, x in 1:w
        r = N0f8(img_cv[y, x, 1])
        g = N0f8(img_cv[y, x, 2])
        b = N0f8(img_cv[y, x, 3])
        img_rgb[y, x] = RGB{N0f8}(r, g, b)
    end
    return img_rgb
end

fig = Figure(resolution=(800,800))
ax = Axis(fig[1,1])

println()

img_rgb = convert_cv_rgb(trunc.(UInt8, img_array./255))
image!(ax, img_rgb)
overlay_points(ax, pts, lbl; marker_size=30)

for i in 1:length(masks)
    overlay_mask(ax, masks[i];)
end

ax.aspect = DataAspect()
fig

#=
parameters = npzread("./predictorTest/testFiles/SP_parameters.npy")
image_size = parameters[1]
target_size = (parameters[2], parameters[3])
num_points = parameters[4]


test_image = npzread("./predictorTest/testFiles/SP_test_image.npy")
test_points = npzread("./predictorTest/testFiles/SP_test_points.npy")
test_labels = npzread("./predictorTest/testFiles/SP_test_labels.npy")
test_box = npzread("./predictorTest/testFiles/SP_test_box.npy")
test_mask = npzread("./predictorTest/testFiles/SP_test_mask.npy")

# Instanziazione

expected_image =
	npzread("./predictorTest/testFiles/SP_set_image_image.npy")
expected_to_pil_image =
	npzread("./predictorTest/testFiles/SP_to_pil_image.npy")
expected_input_image =
	npzread("./predictorTest/testFiles/SP_input_image.npy")
expected_input_image_array =
	npzread("./predictorTest/testFiles/SP_input_image_torch.npy")
expected_input_image_preprocess =
	npzread("./predictorTest/testFiles/SP_input_image_preprocess.npy")
expected_first_x_preprocess =
	npzread("./predictorTest/testFiles/SP_first_x_preprocess.npy")


predictor = SamPredictor(sam)
set_image(predictor, test_image)

println("image size: ", size(test_image))
println("predictor.original_size: ", predictor.original_size)

expected_predictor_features =
	npzread("./predictorTest/testFiles/SP_predictor_features.npy")

is_correct(
	"predictor_features corretto? ",
	predictor.features, expected_predictor_features)  

# get_image_embedding
expected_image_embedding =
	npzread("./predictorTest/testFiles/SP_image_embedding.npy")

image_embedding = get_image_embedding(predictor)

is_correct(
	"image_embedding corretto? ",
	image_embedding, expected_image_embedding)


# predict with points
expected_apply_coords =
	npzread("./predictorTest/testFiles/SP_apply_coords.npy")
expected_coords_array =
	npzread("./predictorTest/testFiles/SP_zero_coords_torch.npy")
expected_labels_array =
	npzread("./predictorTest/testFiles/SP_zero_labels_torch.npy")

expected_transformer_first_coords =
	npzread("./predictorTest/testFiles/SP_transformer_first_coords.npy")
expected_transformer_second_coords =
	npzread("./predictorTest/testFiles/SP_transformer_second_coords.npy")
expected_transformer_third_coords =
	npzread("./predictorTest/testFiles/SP_transformer_third_coords.npy")


expected_predict_sparse_embeddings =
	npzread("./predictorTest/testFiles/SP_predict_sparse_embeddings.npy")
expected_predict_dense_embeddings =
	npzread("./predictorTest/testFiles/SP_predict_dense_embeddings.npy")
expected_predict_low_res_masks =
	npzread("./predictorTest/testFiles/SP_predict_low_res_masks.npy")
expected_predict_iou_predictions =
	npzread("./predictorTest/testFiles/SP_predict_iou_predictions.npy")
expected_predict_maskss =
	npzread("./predictorTest/testFiles/SP_predict_maskss.npy")
expected_predict_self_features =
	npzread("./predictorTest/testFiles/SP_predict_self_features.npy")
expected_predict_get_dense_pe =
	npzread("./predictorTest/testFiles/SP_predict_get_dense_pe.npy")

expected_masks =
	npzread("./predictorTest/testFiles/SP_point_masks.npy")
expected_scores =
	npzread("./predictorTest/testFiles/SP_point_scores.npy")
expected_logits =
	npzread("./predictorTest/testFiles/SP_point_logits.npy")


masks, scores, logits = predict(
	predictor,
	point_coords = test_points,
	point_labels = test_labels,
	multimask_output = true,
	return_logits = false,
)

println(size(masks[1]))
println(count(masks .!= expected_masks), "\tmask diverse")
is_correct("scores corretto? ", scores, expected_scores)
is_correct("logits corretto? ", logits, expected_logits)


# predictor with box
expected_masks_box =
	npzread("./predictorTest/testFiles/SP_box_masks.npy")
expected_scores_box =
	npzread("./predictorTest/testFiles/SP_box_scores.npy")
expected_logits_box =
	npzread("./predictorTest/testFiles/SP_box_logits.npy")


masks_box, scores_box, logits_box = predict(
	predictor,
	box = test_box,
	multimask_output = true,
	return_logits = false,
)

is_correct("masks_box corretto? ", masks_box, expected_masks_box)
is_correct("scores_box corretto? ", scores_box, expected_scores_box)
is_correct("logits_box corretto? ", logits_box, expected_logits_box)




# predictor with mask
expected_masks_mask =
	npzread("./predictorTest/testFiles/SP_box_masks.npy")
expected_scores_mask =
	npzread("./predictorTest/testFiles/SP_box_scores.npy")
expected_logits_mask =
	npzread("./predictorTest/testFiles/SP_box_logits.npy")


masks_mask, scores_mask, logits_mask = predict(
	predictor,
	mask_input = test_mask,
	multimask_output = true,
	return_logits = false,
)

is_correct("masks_mask corretto? ", masks_mask, expected_masks_mask)
is_correct("scores_mask corretto? ", scores_mask, expected_scores_mask)
is_correct("logits_mask corretto? ", logits_mask, expected_logits_mask)
=#

# end of file: test_predictor.jl
# -----------------------------
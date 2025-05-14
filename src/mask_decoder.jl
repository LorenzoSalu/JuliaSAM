using Lux
using Lux: Dense
using Lux: Conv
using CUDA
using Interpolations
using TensorOperations
using Einsum
using Random
using NNlib



########################################################
# MLP:
########################################################

struct MLP
	num_layers::Int
	layers::Chain
	layers_ps::NamedTuple
	layers_st::NamedTuple
	sigmoid_output::Bool
end

function MLP(;
	input_dim::Int,
	hidden_dim::Int,
	output_dim::Int,
	num_layers::Int,
	sigmoid_output::Bool = false,
)

	num_layers = num_layers

	h = fill(hidden_dim, num_layers - 1)

	in_dims = [input_dim; h]
	out_dims = [h; output_dim]

	layers = Chain(
		[Dense(in_dim => out_dim)
		 for (in_dim, out_dim) in zip(in_dims, out_dims)]
	)

	rng = Random.MersenneTwister()
	layers_ps, layers_st = Lux.setup(rng, layers)

	sigmoid_output = sigmoid_output

	return MLP(
		num_layers,
		layers,
		layers_ps,
		layers_st,
		sigmoid_output,
	)
end

function (self::MLP)(x::AbstractArray)
	A, B, C, D = size(x)
	x = sam_reshape(x, (:, D))'

	for i in 1:self.num_layers
		if i < self.num_layers
			tmp, _ = self.layers[i](x, self.layers_ps[i], self.layers_st[i])
			x = relu.(tmp)
		else
			x, _ = self.layers[i](x, self.layers_ps[i], self.layers_st[i])
		end

	end

	if self.sigmoid_output
		x = sigmoid.(x)
	end

	D = size(x, 1)
	x = sam_reshape(x', (A, B, C, D))

	return x
end

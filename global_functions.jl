using SpecialFunctions

function gelu_exact(x)
    return 0.5f0 * x * (1f0 + erf(x / sqrt(2f0)))
end
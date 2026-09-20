struct BinomialSmoother <: Smoother end

Distributions.params(::BinomialSmoother) = (;)

supports_sample_size(::BinomialSmoother, n::Integer) = n >= 1

function κ(::BinomialSmoother, n::Integer, r::Integer, u::Real)
    return Distributions.cdf(Distributions.Beta(r, n + 1 - r), u)
end

function smoothing_distribution(S::BinomialSmoother, n::Integer, r::Integer)
    _check_smoothing_index(n, r)
    _check_smoother(S, n)

    return Distributions.Beta(r, n + 1 - r)
end
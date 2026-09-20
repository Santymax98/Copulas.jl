abstract type Smoother end

Base.broadcastable(S::Smoother) = Ref(S)
Base.eltype(S::Smoother) = _sample_eltype(S)

function κ end
function supports_sample_size end

struct SmoothingDistribution{ST<:Smoother} <: Distributions.ContinuousUnivariateDistribution
    smoother::ST
    n::Int
    r::Int

    function SmoothingDistribution{ST}(S::ST, n::Integer, r::Integer) where {ST<:Smoother}
        _check_smoothing_index(n, r)
        _check_smoother(S, n)

        return new{ST}(S, Int(n), Int(r))
    end
end

function SmoothingDistribution(S::Smoother, n::Integer, r::Integer)
    return SmoothingDistribution{typeof(S)}(S, n, r)
end

Base.minimum(::SmoothingDistribution) = 0.0
Base.maximum(::SmoothingDistribution) = 1.0

function Distributions.cdf(D::SmoothingDistribution, u::Real)
    u <= 0 && return zero(float(u))
    u >= 1 && return one(float(u))

    return κ(D.smoother, D.n, D.r, u)
end

function Distributions.logpdf(D::SmoothingDistribution, u::Real)
    0 <= u <= 1 || return oftype(float(u), -Inf)

    p = ForwardDiff.derivative(x -> Distributions.cdf(D, x), float(u))

    return p > zero(p) ? log(p) : oftype(p, -Inf)
end

Distributions.pdf(D::SmoothingDistribution, u::Real) = exp(Distributions.logpdf(D, u))
Distributions.quantile(D::SmoothingDistribution, p::Real) = _quantile_from_cdf(D, p)
Distributions.rand(rng::Distributions.AbstractRNG, D::SmoothingDistribution) = Distributions.quantile(D, rand(rng))

smoothing_distribution(S::Smoother, n::Integer, r::Integer) = SmoothingDistribution(S, n, r)
smoothing_cdf(S::Smoother, n::Integer, r::Integer, u::Real) = Distributions.cdf(smoothing_distribution(S, n, r), u)
smoothing_pdf(S::Smoother, n::Integer, r::Integer, u::Real) = Distributions.pdf(smoothing_distribution(S, n, r), u)
smoothing_logpdf(S::Smoother, n::Integer, r::Integer, u::Real) = Distributions.logpdf(smoothing_distribution(S, n, r), u)
smoothing_quantile(S::Smoother, n::Integer, r::Integer, p::Real) = Distributions.quantile(smoothing_distribution(S, n, r), p)

@inline function _check_smoothing_index(n::Integer, r::Integer)
    n >= 1 || throw(ArgumentError("the sample size n must be positive; got n=$n"))
    1 <= r <= n || throw(ArgumentError("the rank r must lie in 1:n; got r=$r for n=$n"))
    return nothing
end

function _check_smoother(S::Smoother, n::Integer)
    n >= 1 || throw(ArgumentError("the sample size n must be positive; got n=$n"))
    supports_sample_size(S, n) || throw(DomainError(n, "$(typeof(S)) does not support sample size n=$n",))
    return nothing
end
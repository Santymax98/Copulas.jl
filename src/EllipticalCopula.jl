"""
    EllipticalCopula{d,MT}

Abstract type.

[The description of the API should be here...]

"""
abstract type EllipticalCopula{d,MT} <: Copula{d} end
Base.eltype(C::CT) where CT<:EllipticalCopula = Base.eltype(N(CT)(C.Σ))
function Distributions._rand!(rng::Distributions.AbstractRNG, C::CT, x::AbstractVector{T}) where {T<:Real, CT <: EllipticalCopula}
    Random.rand!(rng,N(CT)(C.Σ),x)
    x .= Distributions.cdf.(U(CT),x)
    return x
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::CT, A::DenseMatrix{T}) where {T<:Real, CT<:EllipticalCopula}
    # More efficient version that precomputes stuff:
    n = N(CT)(C.Σ)
    u = U(CT)
    Random.rand!(rng,n,A)
    A .= Distributions.cdf.(u,A)
    return A
end
function Distributions._logpdf(C::CT, u) where {CT <: EllipticalCopula}
    x = StatsBase.quantile.(U(CT),u)
    return Distributions.logpdf(N(CT)(C.Σ),x) - sum(Distributions.logpdf.(U(CT),x))
end
function make_cor!(Σ)
    # Verify that Σ is a correlation matrix, otherwise make it so : 
    d = size(Σ,1)
    σ = [1/sqrt(Σ[i,i]) for i in 1:d]
    for i in 1:d
        for j in 1:d
            Σ[i,j] *= σ[i] .* σ[j]
        end
    end
end
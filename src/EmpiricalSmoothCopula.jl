struct EmpiricalSmoothCopula{d,ST,CT,RT} <: Copula{d}
    ranks::RT
    smoother::ST
    survival_copula::CT
end

function EmpiricalSmoothCopula{d}(data::AbstractMatrix, smoother::Smoother; survival_copula=IndependentCopula{d}(), pseudo_values::Bool=true,) where {d}
    d >= 2 || throw(ArgumentError("a public copula requires dimension d ≥ 2; got d=$d"))
    size(data, 1) == d || throw(DimensionMismatch("data must have $d rows"))
    survival_copula isa Copula{d} || throw(DimensionMismatch("the smoothing survival copula must have dimension $d",))

    if pseudo_values
        all(0 .<= data .<= 1) || throw(DomainError(data, "pseudo-observations must lie in [0,1]",))
    end

    _require_tie_free_rows(data, "EmpiricalSmoothCopula")

    n = size(data, 2)
    ranks = Matrix{Int}(undef, d, n)

    @inbounds for j in 1:d
        ranks[j, :] = StatsBase.ordinalrank(@view data[j, :])
    end

    return _empirical_smooth_from_ranks(ranks, smoother, survival_copula)
end

EmpiricalSmoothCopula(data::AbstractMatrix, smoother::Smoother; kwargs...,) = EmpiricalSmoothCopula{size(data, 1)}(data, smoother; kwargs...)
EmpiricalSmoothCopula(d::Integer, data::AbstractMatrix, smoother::Smoother; kwargs...,) = EmpiricalSmoothCopula{d}(data, smoother; kwargs...)
Distributions.params(C::EmpiricalSmoothCopula) = (ranks=C.ranks, smoother=C.smoother, survival_copula=C.survival_copula,)

function Base.eltype(C::EmpiricalSmoothCopula)
    return promote_type(eltype(C.smoother), eltype(C.survival_copula))
end

function _cdf(C::EmpiricalSmoothCopula{d}, u) where {d}
    n = size(C.ranks, 2)
    T = promote_type(eltype(u), eltype(C))
    q = Vector{T}(undef, d)
    value = zero(T)

    @inbounds for i in 1:n
        for j in 1:d
            q[j] = smoothing_cdf(C.smoother, n, C.ranks[j, i], u[j],)
        end

        value += Distributions.cdf(C.survival_copula, q)
    end

    return value / n
end

function Distributions._logpdf(C::EmpiricalSmoothCopula{d}, u::AbstractVector) where {d}
    n = size(C.ranks, 2)
    T = promote_type(eltype(u), eltype(C))
    q = Vector{T}(undef, d)
    logdens = T(-Inf)

    @inbounds for i in 1:n
        logterm = zero(T)

        for j in 1:d
            r = C.ranks[j, i]
            q[j] = smoothing_cdf(C.smoother, n, r, u[j])
            logterm += smoothing_logpdf(C.smoother, n, r, u[j])
        end

        logterm += Distributions.logpdf(C.survival_copula, q)
        logdens = LogExpFunctions.logaddexp(logdens, logterm)
    end

    return logdens - log(n)
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::EmpiricalSmoothCopula{d}, A::AbstractMatrix{T}) where {d,T<:Real}
    size(A, 1) == d || throw(ArgumentError("Dimension mismatch between copula and output matrix"))

    n = size(C.ranks, 2)
    m = size(A, 2)

    components = rand(rng, 1:n, m)
    Z = Matrix{T}(undef, d, m)
    Distributions._rand!(rng, C.survival_copula, Z)

    @inbounds for col in axes(A, 2)
        i = components[col]

        for j in 1:d
            r = C.ranks[j, i]
            A[j, col] = smoothing_quantile(C.smoother, n, r, Z[j, col])
        end
    end

    return A
end

function _empirical_smooth_from_ranks(ranks::AbstractMatrix{<:Integer}, smoother::Smoother, survival_copula::Copula{d}) where {d}
    size(ranks, 1) == d || throw(DimensionMismatch("ranks must have $d rows"))
    n = size(ranks, 2)
    _check_smoother(smoother, n)
    return EmpiricalSmoothCopula{d,typeof(smoother),typeof(survival_copula),typeof(ranks)}(ranks, smoother, survival_copula,)
end

function SubsetCopula(C::EmpiricalSmoothCopula{d}, dims::NTuple{p,Int}) where {d,p}
    dims == Tuple(1:d) && return C
    p == 1 && return Distributions.Uniform()

    ranks = C.ranks[collect(dims), :]
    survival_copula = subsetdims(C.survival_copula, dims)

    return _empirical_smooth_from_ranks(ranks, C.smoother, survival_copula)
end

function _partial_cdf(C::EmpiricalSmoothCopula, is, js, uᵢₛ, uⱼₛ)
    n = size(C.ranks, 2)

    function component(i)
        qᵢₛ = ntuple(k -> begin
            j = is[k]
            smoothing_cdf(C.smoother, n, C.ranks[j, i], uᵢₛ[k])
        end, length(is))

        qⱼₛ = ntuple(k -> begin
            j = js[k]
            smoothing_cdf(C.smoother, n, C.ranks[j, i], uⱼₛ[k])
        end, length(js))

        value = _partial_cdf(C.survival_copula, is, js, qᵢₛ, qⱼₛ)

        @inbounds for k in eachindex(js)
            j = js[k]
            value *= smoothing_pdf(C.smoother, n, C.ranks[j, i], uⱼₛ[k])
        end

        return value
    end

    total = component(1)

    @inbounds for i in 2:n
        total += component(i)
    end

    return total / n
end

copula_measure_style(::Type{<:EmpiricalSmoothCopula{d,ST,CT}}) where {d,ST,CT} = copula_measure_style(CT)
copula_measure_style(C::EmpiricalSmoothCopula) = copula_measure_style(C.survival_copula)
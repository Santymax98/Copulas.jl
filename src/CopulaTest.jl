"""
    CopulaTest <: HypothesisTest

Abstract supertype for hypothesis tests concerning copulas.

Concrete subtypes store the observed test statistic, p-value, sample size,
dimension, calibration method, and any test-specific details needed for display
or reproducibility.
"""
abstract type CopulaTest <: HypothesisTest end

"""
    teststatistic(test::CopulaTest)

Return the observed value of the test statistic.
"""
teststatistic(test::CopulaTest) = test.statistic_value

"""
    pvalue(test::CopulaTest)

Return the p-value of `test`.
"""
pvalue(test::CopulaTest) = test.p

StatsBase.nobs(test::CopulaTest) = test.n

function _test_pseudos(U::AbstractMatrix{<:Real}, pseudo_values::Bool)
    all(isfinite, U) || throw(ArgumentError("input data must be finite"))
    V = pseudo_values ? Matrix{Float64}(U) : pseudos(U)
    all(x -> 0 <= x <= 1, V) ||
        throw(ArgumentError("pseudo-observations must lie in [0, 1]"))
    d, n = size(V)
    d >= 2 || throw(ArgumentError("at least two components are required"))
    n >= 2 || throw(ArgumentError("at least two observations are required"))
    return V, d, n
end

function _empirical_copula_partial(Cn::EmpiricalCopula, u::AbstractVector,
        l::Integer, h::Real)
    lo = Vector{Float64}(u)
    hi = Vector{Float64}(u)
    lo[l] = max(lo[l] - h, 0.0)
    hi[l] = min(hi[l] + h, 1.0)
    width = hi[l] - lo[l]
    return width > 0 ? (Distributions.cdf(Cn, hi) - Distributions.cdf(Cn, lo)) / width : 0.0
end

function _multiplier_pvalue(matrices, observed::Real, N::Integer,
        rng::Distributions.AbstractRNG; weights=nothing, scale::Real,
        strict::Bool=false, correction=0.5)
    n = size(first(matrices), 2)
    ξ = Vector{Float64}(undef, n)
    work = Vector{Float64}(undef, n)
    inv_sqrt_n = inv(sqrt(n))
    exceedances = 0

    for _ in 1:N
        Random.randexp!(rng, ξ)
        ξ .-= Statistics.mean(ξ)
        bootstrap_stat = 0.0

        if weights === nothing
            for Q in matrices
                LinearAlgebra.mul!(work, Q, ξ)
                @inbounds for i in 1:n
                    bootstrap_stat += abs2(inv_sqrt_n * work[i])
                end
            end
        else
            for (Q, w) in zip(matrices, weights)
                LinearAlgebra.mul!(work, Q, ξ)
                @inbounds for i in 1:n
                    bootstrap_stat += abs2(inv_sqrt_n * work[i]) * w[i]
                end
            end
        end

        value = scale * bootstrap_stat
        exceedances += strict ? value > observed : value >= observed
    end

    correction === nothing && return exceedances / N
    return (correction + exceedances) / (N + 1)
end

################################################################################
##### Independence
################################################################################

"""
    IndependenceCopulaTest(U; statistic=:cvm, N=1000,
                           calibration=:simulation,
                           pseudo_values=false, rng=Random.default_rng())

Test mutual independence between the components of a random vector.

The default statistic is the Cramer-von Mises functional of the empirical
copula process evaluated against the product copula. The p-value is obtained
by simulating the finite-sample null distribution under mutual independence.

If `pseudo_values=false`, the input is transformed with [`pseudos`](@ref)
before the statistic is computed.
"""
struct IndependenceCopulaTest{S<:Real,P<:Real} <: CopulaTest
    n::Int
    dimension::Int
    statistic_value::S
    p::P
    n_resamples::Int
    statistic::Symbol
    calibration::Symbol
end

function IndependenceCopulaTest(U::AbstractMatrix{<:Real}; statistic::Symbol=:cvm,
        N::Integer=1000, calibration::Symbol=:simulation, pseudo_values::Bool=false,
        rng::Distributions.AbstractRNG=Random.default_rng())
    statistic === :cvm ||
        throw(ArgumentError("Only `statistic=:cvm` is implemented for IndependenceCopulaTest."))
    calibration === :simulation ||
        throw(ArgumentError("Only `calibration=:simulation` is implemented for IndependenceCopulaTest."))
    N >= 1 || throw(ArgumentError("`N` must be positive."))

    V, d, n = _test_pseudos(U, pseudo_values)
    observed = _independence_cvm_statistic(V)
    sample = Matrix{Float64}(undef, d, n)
    exceedances = 0

    for _ in 1:N
        Random.rand!(rng, sample)
        exceedances += _independence_cvm_statistic(pseudos(sample)) >= observed
    end

    p = (0.5 + exceedances) / (N + 1)
    return IndependenceCopulaTest(n, d, observed, p, Int(N), statistic, calibration)
end

function _independence_cvm_statistic(U::AbstractMatrix)
    Cn = EmpiricalCopula(U; pseudo_values=true)
    s = 0.0
    @inbounds for u in eachcol(U)
        s += abs2(Distributions.cdf(Cn, u) - prod(u))
    end
    return s
end

################################################################################
##### Exchangeability
################################################################################

"""
    ExchangeabilityCopulaTest(U; statistic=:Sn, permutations=:G2,
                              weight=:wm2, N=1000,
                              calibration=:multiplier,
                              pseudo_values=false, rng=Random.default_rng())

Test exchangeability of a copula in arbitrary dimension.

The default statistic is the empirical-copula integral `Sn` of Harder and
Stadtmuller. With `permutations=:G2`, dimension two uses the single
transposition `(12)`, while higher dimensions use `(12)` and the left shift
`(12...d)`. Approximate p-values are obtained with an exponential multiplier
bootstrap.
"""
struct ExchangeabilityCopulaTest{S<:Real,P<:Real,D} <: CopulaTest
    n::Int
    dimension::Int
    statistic_value::S
    p::P
    n_resamples::Int
    statistic::Symbol
    calibration::Symbol
    details::D
end

ExchangeabilityCopulaTest(n::Integer, dimension::Integer, statistic_value::Real,
    p::Real, n_resamples::Integer, statistic::Symbol, calibration::Symbol) =
    ExchangeabilityCopulaTest(Int(n), Int(dimension), statistic_value, p,
        Int(n_resamples), statistic, calibration, (;))

function ExchangeabilityCopulaTest(U::AbstractMatrix{<:Real}; statistic::Symbol=:Sn,
        permutations=:G2, weight::Symbol=:wm2, N::Integer=1000,
        calibration::Symbol=:multiplier, pseudo_values::Bool=false,
        rng::Distributions.AbstractRNG=Random.default_rng())
    statistic === :Sn ||
        throw(ArgumentError("Only `statistic=:Sn` is implemented for ExchangeabilityCopulaTest."))
    calibration === :multiplier ||
        throw(ArgumentError("Only `calibration=:multiplier` is implemented for ExchangeabilityCopulaTest."))
    N >= 1 || throw(ArgumentError("`N` must be positive."))

    V, d, n = _test_pseudos(U, pseudo_values)
    perms = _exchangeability_permutations(permutations, d)
    observed = _exchangeability_sn_statistic(V, perms, weight)
    p, bandwidth = _exchangeability_multiplier_pvalue(V, perms, weight, observed, N, rng)
    details = (; permutations, generator=perms, weight,
        multiplier=:exponential, derivative_bandwidth=bandwidth)
    return ExchangeabilityCopulaTest(n, d, observed, p, Int(N), statistic, calibration, details)
end

function _exchangeability_permutations(permutations, d::Integer)
    identity_perm = ntuple(i -> i, d)
    raw = if permutations === :G2
        d == 2 ? ((2, 1),) :
        ((2, 1, ntuple(i -> i + 2, d - 2)...), ntuple(i -> i == d ? 1 : i + 1, d))
    elseif permutations === :G1
        ntuple(i -> Tuple(j == 1 ? i + 1 : j == i + 1 ? 1 : j for j in 1:d), d - 1)
    elseif permutations === :all
        Combinatorics.permutations(1:d)
    else
        is_single = (permutations isa Tuple || permutations isa AbstractVector) &&
            length(permutations) == d && all(x -> x isa Integer, permutations)
        is_single ? (permutations,) : permutations
    end

    result = NTuple{d,Int}[]
    for perm in raw
        p = Tuple(Int.(perm))
        length(p) == d || throw(ArgumentError("permutations must have length $d"))
        sort(collect(p)) == collect(1:d) || throw(ArgumentError("invalid permutation `$perm`"))
        p == identity_perm || push!(result, p)
    end
    isempty(result) && throw(ArgumentError("at least one non-identity permutation is required"))
    return Tuple(result)
end

function _exchangeability_weight(u::AbstractVector, perm::Tuple, weight::Symbol)
    weight === :none && return 1.0
    weight === :wm2 ||
        throw(ArgumentError("Only `weight=:wm2` and `weight=:none` are implemented."))

    m = minimum(u)
    omega = if count(i -> perm[i] != i, eachindex(perm)) == 2 &&
            all(perm[perm[i]] == i for i in eachindex(perm))
        i = findfirst(k -> perm[k] != k, eachindex(perm))
        j = perm[i]
        abs(u[i] - u[j])
    else
        v = sort(collect(u))
        sum(v[i] - m for i in cld(length(v), 2) + 1:length(v))
    end
    wm = min(m, omega, length(u) - 1 + m - sum(u))
    return abs2(max(wm, 0.0))
end

function _exchangeability_sn_statistic(U::AbstractMatrix, permutations, weight::Symbol)
    d, n = size(U)
    Cn = EmpiricalCopula(U; pseudo_values=true)
    s = 0.0
    uperm = Vector{Float64}(undef, d)

    @inbounds for perm in permutations
        for i in 1:n
            u = @view U[:, i]
            for k in 1:d
                uperm[k] = u[perm[k]]
            end
            diff = Distributions.cdf(Cn, u) - Distributions.cdf(Cn, uperm)
            s += abs2(diff) * _exchangeability_weight(u, perm, weight)
        end
    end
    return s / n
end

function _exchangeability_multiplier_matrices(U::AbstractMatrix, permutations, weight::Symbol)
    d, n = size(U)
    Cn = EmpiricalCopula(U; pseudo_values=true)
    h = inv(sqrt(n))
    partials = Matrix{Float64}(undef, d, n)
    q_matrices = Matrix{Float64}[]
    weights = Vector{Float64}[]

    @inbounds for i in 1:n
        u = @view U[:, i]
        for l in 1:d
            partials[l, i] = _empirical_copula_partial(Cn, u, l, h)
        end
    end

    @inbounds for perm in permutations
        invperm = Vector{Int}(undef, d)
        for k in 1:d
            invperm[perm[k]] = k
        end

        Q = Matrix{Float64}(undef, n, n)
        w = Vector{Float64}(undef, n)
        for i in 1:n
            u = @view U[:, i]
            w[i] = _exchangeability_weight(u, perm, weight)
            for j in 1:n
                le_u = true
                le_up = true
                for k in 1:d
                    U[k, j] <= u[k] || (le_u = false)
                    U[k, j] <= u[perm[k]] || (le_up = false)
                end

                q = (le_u ? 1.0 : 0.0) - (le_up ? 1.0 : 0.0)
                for l in 1:d
                    le_margin = U[l, j] <= u[l]
                    le_permuted_margin = U[invperm[l], j] <= u[l]
                    q -= partials[l, i] *
                        ((le_margin ? 1.0 : 0.0) - (le_permuted_margin ? 1.0 : 0.0))
                end
                Q[i, j] = q
            end
        end
        push!(q_matrices, Q)
        push!(weights, w)
    end
    return q_matrices, weights, h
end

function _exchangeability_multiplier_pvalue(U::AbstractMatrix, permutations, weight::Symbol,
        observed::Real, N::Integer, rng::Distributions.AbstractRNG)
    _, n = size(U)
    q_matrices, weights, h = _exchangeability_multiplier_matrices(U, permutations, weight)
    p = _multiplier_pvalue(q_matrices, observed, N, rng;
        weights, scale=inv(n^2), strict=true, correction=nothing)
    return p, h
end

################################################################################
##### Radial Symmetry
################################################################################

"""
    RadialSymmetryCopulaTest(U; statistic=:Sn, N=1000,
                             calibration=:randomization,
                             pseudo_values=false, rng=Random.default_rng())

Test radial symmetry of a copula.

The default statistic is the empirical-copula Cramer-von Mises statistic `Sn`,
comparing the empirical copula with its survival counterpart. The p-value is
obtained by randomly reflecting observations as `u` or `1 - u` under the null
hypothesis.
"""
struct RadialSymmetryCopulaTest{S<:Real,P<:Real,D} <: CopulaTest
    n::Int
    dimension::Int
    statistic_value::S
    p::P
    n_resamples::Int
    statistic::Symbol
    calibration::Symbol
    details::D
end

RadialSymmetryCopulaTest(n::Integer, dimension::Integer, statistic_value::Real,
    p::Real, n_resamples::Integer, statistic::Symbol, calibration::Symbol) =
    RadialSymmetryCopulaTest(Int(n), Int(dimension), statistic_value, p,
        Int(n_resamples), statistic, calibration, (;))

function RadialSymmetryCopulaTest(U::AbstractMatrix{<:Real}; statistic::Symbol=:Sn,
        N::Integer=1000, calibration::Symbol=:randomization, pseudo_values::Bool=false,
        rng::Distributions.AbstractRNG=Random.default_rng())
    statistic === :Sn ||
        throw(ArgumentError("Only `statistic=:Sn` is implemented for RadialSymmetryCopulaTest."))
    calibration === :randomization ||
        throw(ArgumentError("Only `calibration=:randomization` is implemented for RadialSymmetryCopulaTest."))
    N >= 1 || throw(ArgumentError("`N` must be positive."))

    V, d, n = _test_pseudos(U, pseudo_values)
    observed = _radial_symmetry_sn_statistic(V)
    sample = similar(V)
    exceedances = 0

    for _ in 1:N
        @inbounds for i in 1:n
            reflected = rand(rng) < 0.5
            for j in 1:d
                sample[j, i] = reflected ? 1 - V[j, i] : V[j, i]
            end
        end
        exceedances += _radial_symmetry_sn_statistic(pseudos(sample)) >= observed
    end

    p = (0.5 + exceedances) / (N + 1)
    details = (; reflection_probability=0.5,)
    return RadialSymmetryCopulaTest(n, d, observed, p, Int(N),
        statistic, calibration, details)
end

function _radial_symmetry_sn_statistic(U::AbstractMatrix)
    Cn = EmpiricalCopula(U; pseudo_values=true)
    Cbar = EmpiricalCopula(1 .- U; pseudo_values=true)
    s = 0.0
    @inbounds for u in eachcol(U)
        s += abs2(Distributions.cdf(Cn, u) - Distributions.cdf(Cbar, u))
    end
    return s / size(U, 2)
end

################################################################################
##### Extreme Value
################################################################################

"""
    ExtremeValueCopulaTest(U; statistic=:Sn, powers=3:5, N=1000,
                           calibration=:multiplier,
                           pseudo_values=false, rng=Random.default_rng())

Test whether a copula belongs to the extreme-value class.

The default statistic is based on max-stability: an extreme-value copula
satisfies `C(u .^ (1 / r))^r == C(u)`. The `powers` argument selects the
max-stability powers `r`. Approximate p-values are obtained with an
exponential multiplier bootstrap.
"""
struct ExtremeValueCopulaTest{S<:Real,P<:Real,D} <: CopulaTest
    n::Int
    dimension::Int
    statistic_value::S
    p::P
    n_resamples::Int
    statistic::Symbol
    calibration::Symbol
    details::D
end

ExtremeValueCopulaTest(n::Integer, dimension::Integer, statistic_value::Real,
    p::Real, n_resamples::Integer, statistic::Symbol, calibration::Symbol) =
    ExtremeValueCopulaTest(Int(n), Int(dimension), statistic_value, p,
        Int(n_resamples), statistic, calibration, (;))

function ExtremeValueCopulaTest(U::AbstractMatrix{<:Real}; statistic::Symbol=:Sn,
        powers=3:5, N::Integer=1000, calibration::Symbol=:multiplier,
        pseudo_values::Bool=false, rng::Distributions.AbstractRNG=Random.default_rng())
    statistic === :Sn ||
        throw(ArgumentError("Only `statistic=:Sn` is implemented for ExtremeValueCopulaTest."))
    calibration === :multiplier ||
        throw(ArgumentError("Only `calibration=:multiplier` is implemented for ExtremeValueCopulaTest."))
    N >= 1 || throw(ArgumentError("`N` must be positive."))

    V, d, n = _test_pseudos(U, pseudo_values)
    r = _max_stability_powers(powers)
    observed = _extreme_value_sn_statistic(V, r)
    p, bandwidth = _extreme_value_multiplier_pvalue(V, r, observed, N, rng)
    details = (; powers=r, multiplier=:exponential, derivative_bandwidth=bandwidth)
    return ExtremeValueCopulaTest(n, d, observed, p, Int(N), statistic, calibration, details)
end

function _max_stability_powers(powers)
    raw = powers isa Real ? (powers,) : Tuple(powers)
    result = Float64[]
    for r in raw
        isfinite(r) && r > 1 ||
            throw(ArgumentError("max-stability powers must be finite and greater than one"))
        push!(result, Float64(r))
    end
    isempty(result) && throw(ArgumentError("at least one max-stability power is required"))
    return Tuple(result)
end

function _extreme_value_sn_statistic(U::AbstractMatrix, powers)
    d, n = size(U)
    Cn = EmpiricalCopula(U; pseudo_values=true)
    uroot = Vector{Float64}(undef, d)
    s = 0.0

    @inbounds for r in powers
        invr = inv(r)
        for u in eachcol(U)
            for k in 1:d
                uroot[k] = u[k]^invr
            end
            diff = Distributions.cdf(Cn, uroot)^r - Distributions.cdf(Cn, u)
            s += abs2(diff)
        end
    end
    return s / n
end

function _extreme_value_multiplier_matrices(U::AbstractMatrix, powers)
    d, n = size(U)
    Cn = EmpiricalCopula(U; pseudo_values=true)
    h = inv(sqrt(n))
    uroot = Vector{Float64}(undef, d)
    partials_u = Vector{Float64}(undef, d)
    partials_root = Vector{Float64}(undef, d)
    matrices = Matrix{Float64}[]

    @inbounds for r in powers
        Q = Matrix{Float64}(undef, n, n)
        invr = inv(r)
        for i in 1:n
            u = @view U[:, i]
            for k in 1:d
                uroot[k] = u[k]^invr
                partials_u[k] = _empirical_copula_partial(Cn, u, k, h)
            end
            croot = Distributions.cdf(Cn, uroot)
            factor = r * croot^(r - 1)
            for k in 1:d
                partials_root[k] = _empirical_copula_partial(Cn, uroot, k, h)
            end

            for j in 1:n
                le_u = true
                le_root = true
                for k in 1:d
                    U[k, j] <= u[k] || (le_u = false)
                    U[k, j] <= uroot[k] || (le_root = false)
                end

                q_u = le_u ? 1.0 : 0.0
                q_root = le_root ? 1.0 : 0.0
                for k in 1:d
                    q_u -= partials_u[k] * (U[k, j] <= u[k] ? 1.0 : 0.0)
                    q_root -= partials_root[k] * (U[k, j] <= uroot[k] ? 1.0 : 0.0)
                end
                Q[i, j] = factor * q_root - q_u
            end
        end
        push!(matrices, Q)
    end
    return matrices, h
end

function _extreme_value_multiplier_pvalue(U::AbstractMatrix, powers, observed::Real,
        N::Integer, rng::Distributions.AbstractRNG)
    _, n = size(U)
    matrices, h = _extreme_value_multiplier_matrices(U, powers)
    p = _multiplier_pvalue(matrices, observed, N, rng;
        scale=inv(n^2), strict=false, correction=0.5)
    return p, h
end

################################################################################
##### Goodness of Fit
################################################################################

"""
    GOFCopulaTest(C, U; statistic=:Sn, N=1000,
                  calibration=:parametric_bootstrap,
                  pseudo_values=false, rng=Random.default_rng())
    GOFCopulaTest(model, U; statistic=:Sn, N=1000,
                  calibration=:parametric_bootstrap,
                  pseudo_values=false, rng=Random.default_rng())
    GOFCopulaTest(model; kwargs...)

Test goodness of fit for a copula or fitted copula model.

The field `hypothesis` distinguishes `:simple`, `:composite`, and `:selected`.
The default statistic is the empirical-copula Cramer-von Mises distance. The
default calibration is parametric bootstrap.
"""
struct GOFCopulaTest{M,S<:Real,P<:Real} <: CopulaTest
    model::M
    hypothesis::Symbol
    n::Int
    dimension::Int
    statistic_value::S
    p::P
    n_resamples::Int
    statistic::Symbol
    calibration::Symbol
end

function GOFCopulaTest(C::Copula, U::AbstractMatrix{<:Real}; statistic::Symbol=:Sn,
        N::Integer=1000, calibration::Symbol=:parametric_bootstrap,
        pseudo_values::Bool=false, rng::Distributions.AbstractRNG=Random.default_rng())
    statistic === :Sn ||
        throw(ArgumentError("Only `statistic=:Sn` is implemented for GOFCopulaTest."))
    calibration === :parametric_bootstrap ||
        throw(ArgumentError("Only `calibration=:parametric_bootstrap` is implemented for GOFCopulaTest."))
    N >= 1 || throw(ArgumentError("`N` must be positive."))

    V, d, n = _test_pseudos(U, pseudo_values)
    length(C) == d || throw(DimensionMismatch("copula dimension does not match input data"))
    observed = _gof_sn_statistic(V, C)
    p = _gof_parametric_bootstrap(C, V, observed, N, rng)
    return GOFCopulaTest(C, :simple, n, d, observed, p, Int(N), statistic, calibration)
end

function GOFCopulaTest(M::CopulaModel, U::AbstractMatrix{<:Real}; statistic::Symbol=:Sn,
        N::Integer=1000, calibration::Symbol=:parametric_bootstrap,
        pseudo_values::Bool=false, rng::Distributions.AbstractRNG=Random.default_rng())
    statistic === :Sn ||
        throw(ArgumentError("Only `statistic=:Sn` is implemented for GOFCopulaTest."))
    calibration === :parametric_bootstrap ||
        throw(ArgumentError("Only `calibration=:parametric_bootstrap` is implemented for GOFCopulaTest."))
    N >= 1 || throw(ArgumentError("`N` must be positive."))

    V, d, n = _test_pseudos(U, pseudo_values)
    C = _copula_of(M)
    length(C) == d || throw(DimensionMismatch("model dimension does not match input data"))
    observed = _gof_sn_statistic(V, C)
    p = _gof_parametric_bootstrap(M, V, observed, N, rng)
    hypothesis = get(M.method_details, :selection, false) ? :selected : :composite
    return GOFCopulaTest(M, hypothesis, n, d, observed, p, Int(N), statistic, calibration)
end

function GOFCopulaTest(M::CopulaModel; kwargs...)
    haskey(M.method_details, :U) ||
        throw(ArgumentError("the fitted model does not store pseudo-observations"))
    return GOFCopulaTest(M, M.method_details.U; pseudo_values=true, kwargs...)
end

function _gof_sn_statistic(U::AbstractMatrix, C::Copula)
    Cn = EmpiricalCopula(U; pseudo_values=true)
    s = 0.0
    @inbounds for u in eachcol(U)
        s += abs2(Distributions.cdf(Cn, u) - Distributions.cdf(C, u))
    end
    return s / size(U, 2)
end

function _gof_refit(M::CopulaModel, U::AbstractMatrix)
    md = M.method_details
    if get(md, :selection, false)
        return Distributions.fit(CopulaModel, Copula, U;
            candidates=md.candidates,
            criterion=md.criterion,
            method=md.requested_method,
            quick_fit=false,
            derived_measures=false,
            vcov=false,
            md.selection_options...)
    end

    return Distributions.fit(CopulaModel, typeof(_copula_of(M)), U;
        method=M.method, quick_fit=false, derived_measures=false, vcov=false)
end

function _gof_parametric_bootstrap(C::Copula, U::AbstractMatrix, observed::Real,
        N::Integer, rng::Distributions.AbstractRNG)
    _, n = size(U)
    exceedances = 0
    for _ in 1:N
        sample = pseudos(rand(rng, C, n))
        exceedances += _gof_sn_statistic(sample, C) >= observed
    end
    return (0.5 + exceedances) / (N + 1)
end

function _gof_parametric_bootstrap(M::CopulaModel, U::AbstractMatrix, observed::Real,
        N::Integer, rng::Distributions.AbstractRNG)
    _, n = size(U)
    C = _copula_of(M)
    exceedances = 0
    for _ in 1:N
        sample = pseudos(rand(rng, C, n))
        bootstrap_model = _gof_refit(M, sample)
        exceedances += _gof_sn_statistic(sample, _copula_of(bootstrap_model)) >= observed
    end
    return (0.5 + exceedances) / (N + 1)
end

"""
    testname(test::CopulaTest)

Return the display name of a copula hypothesis test.
"""
testname(::IndependenceCopulaTest) = "Copula independence test"
testname(::ExchangeabilityCopulaTest) = "Copula exchangeability test"
testname(::RadialSymmetryCopulaTest) = "Copula radial symmetry test"
testname(::ExtremeValueCopulaTest) = "Extreme-value copula test"
testname(::GOFCopulaTest) = "Copula goodness-of-fit test"

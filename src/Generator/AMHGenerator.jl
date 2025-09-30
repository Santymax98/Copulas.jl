"""
    AMHGenerator{T}

Fields:
  - θ::Real - parameter

Constructor

    AMHGenerator(θ)
    AMHCopula(d,θ)

The [AMH](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [-1,1)``. It is an Archimedean copula with generator:

```math
\\phi(t) = 1 - \\frac{1-\\theta}{e^{-t} - \\theta}.
```

Special cases:
- When θ = 0, it is the IndependentCopula

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct AMHGenerator{T} <: Generator
    θ::T
    function AMHGenerator(θ)
        if (θ < -1) || (θ > 1)
            throw(ArgumentError("Theta must be in [-1,1), you provided $θ."))
        elseif θ == 0
            return IndependentGenerator()
        else
            θ, _ = promote(θ, 1.0)
            return new{typeof(θ)}(θ)
        end
    end
end
const AMHCopula{d, T} = ArchimedeanCopula{d, AMHGenerator{T}}
AMHCopula(d, θ) = ArchimedeanCopula(d, AMHGenerator(θ))
Distributions.params(G::AMHGenerator) = (G.θ,)


function _find_critical_value_amh(k; step=1e-7)
    # Return the threshold θ_k such that “θ < θ_k ⇒ max_monotony returns k-1”.
    # This unifies analytic and numeric thresholds and falls back to a
    # numerical search via PolyLog for large k.
    k == 2  && return -1.0
    k == 3  && return sqrt(3) - 2
    k == 4  && return -5 + 2*sqrt(6)
    k == 5  && return -13/2 - sqrt(105)/2 + (sqrt(2)/2) * sqrt(13*sqrt(105) + 135)
    k == 6  && return -14 - 3 * sqrt(15) + sqrt(6) * sqrt(14 * sqrt(15) + 55)
    k == 7  && return -0.00914869999999993
    k == 8  && return -0.004376199999998468
    k == 9  && return -0.002121400000000042
    k == 10 && return -0.0010375999999997928
    k == 11 && return -0.0005105999999999994
    k == 12 && return -0.00025240000000000527
    k == 13 && return -0.0001252000000000022
    k == 14 && return -6.220000000000067e-5
    k == 15 && return -3.099999999999991e-5
    k == 16 && return -1.5500000000000048e-5
    k == 17 && return -7.699999999999994e-6
    k == 18 && return -3.839999999999973e-6
    k == 19 && return -1.9199999999999918e-6
    k == 20 && return -9.600000000000008e-7

    x = 0.0
    while x > -1
        PolyLog.reli.(-k, x) > 0 && break
        x -= step
    end
    return x
end

function max_monotony(G::AMHGenerator)
    G.θ >= 0 && return Inf
    @inbounds for k in 3:100
        if G.θ < _find_critical_value_amh(k, step=1e-7)
            return k - 1
        end
    end
    return 100
end
_θ_bounds(::Type{<:AMHGenerator}, d::Integer) = (_find_critical_value_amh(d, step=1e-7), 1.0)

ϕ(  G::AMHGenerator, t) = (1-G.θ)/(exp(t)-G.θ)
ϕ⁻¹(G::AMHGenerator, t) = log(G.θ + (1-G.θ)/t)
ϕ⁽¹⁾(G::AMHGenerator, t) = -((1-G.θ) * exp(t)) / (exp(t) - G.θ)^2
ϕ⁽ᵏ⁾(G::AMHGenerator, ::Val{k}, t) where k = (-1)^k * (1 - G.θ) / G.θ * PolyLog.reli(-k, G.θ * exp(-t))
ϕ⁻¹⁽¹⁾(G::AMHGenerator, t) = (G.θ - 1) / (G.θ * (t - 1) * t + t)
williamson_dist(G::AMHGenerator, ::Val{d}) where d = G.θ >= 0 ? WilliamsonFromFrailty(1 + Distributions.Geometric(1-G.θ),Val{d}()) : WilliamsonTransforms.𝒲₋₁(t -> ϕ(G,t),Val{d}())
frailty(G::AMHGenerator) = G.θ >= 0 ? Distributions.Geometric(1-G.θ) : throw("No frailty exists for AMH when θ < 0")
function _amh_tau(θ)
    if abs(θ) < 0.01
        return 2/9  * θ
            + 1/18  * θ^2
            + 1/45  * θ^3
            + 1/90  * θ^4
            + 2/315 * θ^5
            + 1/252 * θ^6
            + 1/378 * θ^7
            + 1/540 * θ^8
            + 2/1485 * θ^9
            + 1/990 * θ^10
    end
    if iszero(θ)
        return zero(θ)
    end
    u = isone(θ) ? θ : θ + (1-θ)^2 * log1p(-θ)
    return 1 - (2/3)*u/θ^2
end
τ(G::AMHGenerator) = _amh_tau(G.θ)
function τ⁻¹(::Type{T},tau) where T<:AMHGenerator
    if tau == zero(tau)
        return tau
    elseif tau > 1/3
        @info "AMHCopula cannot handle κ > 1/3."
        return one(tau)
    elseif tau < (5 - 8*log(2))/3
        @info "AMHCopula cannot handle κ < 5 - 8ln(2))/3 (approx -0.1817)."
        return -one(tau)
    end
    search_range = tau > 0 ? (0,1) : (-1,0)
    return Roots.find_zero(θ -> tau - _amh_tau(θ), search_range)
end

function ρ(G::AMHGenerator)
    # Taken from https://cran.r-project.org/web/packages/copula/vignettes/rhoAMH-dilog.pdf
    a = G.θ
    if isnan(a)
        return a
    end
    aa = abs(a)
    if aa < 7e-16
        return a / 3
    elseif aa < 1e-4
        return a / 3 * (1 + a / 4)
    elseif aa < 0.002
        return a * (1/3 + a * (1/12 + a * 3/100))
    elseif aa < 0.007
        return a * (1/3 + a * (1/12 + a * (3/100 + a / 75)))
    elseif aa < 0.016
        return a * (1/3 + a * (1/12 + a * (3/100 + a * (1/75 + a / 147))))
    else
        term1 = 3 / a * (4 * (1 + 1 / a) * PolyLog.reli2(a))
        term2 = if a < 1
            8 * (1 / a - 1) * log1p(-a)
        else
            0.0
        end
        return term1 - term2 - (a + 12)
    end
end

# Inversa de Spearman para AMH vía Brent + fallback a Bisection
function ρ⁻¹(::Type{AMHGenerator}, ρ̂::Real; xtol=1e-12, atol=1e-12, maxevals=10_000)
    # Casos triviales
    if !isfinite(ρ̂)
        return oftype(ρ̂, NaN)
    end
    if ρ̂ == 0
        return zero(ρ̂)
    end

    # Dominio de AMH: θ ∈ (-1, 1)
    Tρ = float(promote_type(typeof(ρ̂), Float64))
    ϵ  = Tρ(1e-12)
    aL = Tρ(-1 + ϵ)
    aU = Tρ( 1 - ϵ)

    # Rango numérico alcanzable por ρ(θ) en ±(1-ε)
    ρL = ρ(AMHGenerator(aL))
    ρU = ρ(AMHGenerator(aU))
    if ρL > ρU
        ρL, ρU = ρU, ρL
        aL, aU = aU, aL
    end

    # Saturaciones (si el objetivo cae fuera del rango numérico)
    if ρ̂ ≤ ρL
        return aL
    elseif ρ̂ ≥ ρU
        return aU
    end

    # Ecuación a resolver: f(θ) = ρ(θ) - ρ̂
    f(θ) = ρ(AMHGenerator(θ)) - ρ̂

    # Brent con bracket global seguro, y fallback a bisección si hay excepción
    try
        return Roots.find_zero(f, (aL, aU), Roots.Brent(); xtol=xtol, atol=atol, maxevals=maxevals)
    catch
        return Roots.find_zero(f, (aL, aU), Roots.Bisection(); xtol=xtol, atol=atol, maxevals=maxevals)
    end
end
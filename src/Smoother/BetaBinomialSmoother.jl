struct BetaBinomialSmoother{T<:Real} <: Smoother
    ρ::T

    function BetaBinomialSmoother(ρ::Real)
        ρ > 1 || throw(DomainError(ρ, "ρ must be greater than 1"))
        ρ = float(ρ)

        return new{typeof(ρ)}(ρ)
    end
end

Distributions.params(S::BetaBinomialSmoother) = (; ρ=S.ρ)
supports_sample_size(S::BetaBinomialSmoother, n::Integer) = 1 < S.ρ < n

function κ(S::BetaBinomialSmoother, n::Integer, r::Integer, u::Real)
    u <= 0 && return zero(u + S.ρ)
    u >= 1 && return one(u + S.ρ)

    θ = (n - S.ρ) / (S.ρ - 1)
    α = u * θ
    β = (1 - u) * θ

    return Distributions.ccdf(Distributions.BetaBinomial(n, α, β), r - 1,)
end
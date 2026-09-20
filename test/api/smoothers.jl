struct _IncompleteSmoother <: Copulas.Smoother end

Copulas.κ(::_IncompleteSmoother, n::Integer, r::Integer, u::Real) = u
Distributions.params(::_IncompleteSmoother) = (;)

struct _MinimalSmoother <: Copulas.Smoother end

Distributions.params(::_MinimalSmoother) = (;)

Copulas.supports_sample_size(::_MinimalSmoother, n::Integer) = n >= 1

function Copulas.κ(::_MinimalSmoother, n::Integer, r::Integer, u::Real)
    return sum(binomial(n, k) * u^k * (1 - u)^(n - k) for k in r:n)
end

@testset "Smoother public extension contract" begin
    incomplete = _IncompleteSmoother()

    @test_throws MethodError Copulas.supports_sample_size(incomplete, 10)
    @test_throws MethodError Copulas.smoothing_distribution(incomplete, 10, 1)

    S = _MinimalSmoother()
    @test Copulas.supports_sample_size(S, 10)

    D = Copulas.smoothing_distribution(S, 10, 3)
    B = Beta(3, 8)

    @test cdf(D, 0.37) ≈ cdf(B, 0.37) atol=1e-12
    @test pdf(D, 0.37) ≈ pdf(B, 0.37) atol=1e-10
    @test quantile(D, 0.5) ≈ quantile(B, 0.5) atol=1e-8
end

@testset "Built-in smoother support" begin
    S = Copulas.BinomialSmoother()

    @test Copulas.supports_sample_size(S, 1)
    @test Copulas.supports_sample_size(S, 10)

    Sbb = Copulas.BetaBinomialSmoother(4.0)

    @test !Copulas.supports_sample_size(Sbb, 4)
    @test Copulas.supports_sample_size(Sbb, 5)
    @test Copulas.supports_sample_size(Sbb, 30)

    @test_throws DomainError Copulas.BetaBinomialSmoother(1.0)
end
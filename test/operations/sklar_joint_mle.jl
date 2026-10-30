
# Joint maximum-likelihood fitting for Sklar distributions.
# Numerical comparisons use fixed random seeds and independent references.

@testset "Sklar joint maximum likelihood" begin

    @testset "Clayton with continuous margins" begin
        rng = MersenneTwister(2026)
        T = SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}
        S = SklarDist(ClaytonCopula{2}(1.5), (Normal(), Exponential()))
        X = rand(rng, S, 150)

        M_ifm = fit(CopulaModel, T, X; sklar_method=:ifm)
        M_jmle = fit(CopulaModel, T, X; sklar_method=:jmle)
        fitted = fitted_distribution(M_jmle)

        @test fitted isa SklarDist
        @test Copulas.fitting_method(M_jmle) === :jmle
        @test StatsBase.dof(M_jmle) == 4
        @test StatsBase.nobs(M_jmle) == 150
        @test isfinite(loglikelihood(M_jmle))
        @test loglikelihood(M_jmle) >= loglikelihood(M_ifm) - 1e-6
        @test all(isfinite, StatsBase.coef(M_jmle))

        direct = fit(T, X; sklar_method=:jmle)
        @test direct isa SklarDist
        @test isapprox(loglikelihood(direct, X), loglikelihood(M_jmle); atol=1e-6)
    end

    @testset "Independent copula oracle" begin
        rng = MersenneTwister(2026)
        T = SklarDist{IndependentCopula,Tuple{Normal,Exponential}}
        S = SklarDist(IndependentCopula{2}(), (Normal(1.0, 1.5), Exponential(2.0)))
        X = rand(rng, S, 150)

        M = fit(CopulaModel, T, X; sklar_method=:jmle)
        m1 = fit_mle(Normal, vec(X[1, :]))
        m2 = fit_mle(Exponential, vec(X[2, :]))
        reference = sum(logpdf(m1, X[1, j]) + logpdf(m2, X[2, j]) for j in axes(X, 2))

        @test fitted_distribution(M).C isa IndependentCopula
        @test StatsBase.dof(M) == 3
        @test isapprox(loglikelihood(M), reference; atol=1e-4)
    end

    @testset "Gamma margin with numerical gradients" begin
        rng = MersenneTwister(2026)
        T = SklarDist{ClaytonCopula,Tuple{Normal,Exponential,Gamma}}
        S = SklarDist(ClaytonCopula{3}(1.2), (Normal(), Exponential(), Gamma(2.5, 1.4)))
        X = rand(rng, S, 120)

        M_ifm = fit(CopulaModel, T, X; sklar_method=:ifm)
        M_jmle = fit(CopulaModel, T, X; sklar_method=:jmle)

        @test Copulas.fitting_method(M_jmle) === :jmle
        @test StatsBase.dof(M_jmle) == 6
        @test isfinite(loglikelihood(M_jmle))
        @test loglikelihood(M_jmle) >= loglikelihood(M_ifm) - 1e-6
    end

    @testset "Weights and estimator replay" begin
        rng = MersenneTwister(2026)
        T = SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}
        S = SklarDist(ClaytonCopula{2}(1.2), (Normal(0.4, 1.1), Exponential(1.7)))
        X = rand(rng, S, 90)

        w = repeat([0.0, 1.0, 2.0], 30)
        idx = vcat((fill(j, Int(w[j])) for j in eachindex(w))...)
        Xrep = X[:, idx]

        Mw = fit(CopulaModel, T, X; sklar_method=:jmle, weights=w)
        Mr = fit(CopulaModel, T, Xrep; sklar_method=:jmle)
        replay = Copulas._refit(Mw, Xrep; replay_input=true)

        @test isapprox(loglikelihood(Mw), loglikelihood(Mr); atol=1e-3)
        @test isapprox(loglikelihood(replay), loglikelihood(Mr); atol=1e-3)
        @test Copulas.fitting_method(replay) === :jmle
    end

    @testset "Bootstrap integration" begin
        rng = MersenneTwister(2026)
        T = SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}
        S = SklarDist(ClaytonCopula{2}(1.5), (Normal(), Exponential()))
        X = rand(rng, S, 150)

        M = fit(CopulaModel, T, X; sklar_method=:jmle)
        I = infer(M; method=:bootstrap, nresamples=4, rng=MersenneTwister(2027))
        V = StatsBase.vcov(I)

        @test size(V) == (4, 4)
        @test all(isfinite, V)
        @test isapprox(V, transpose(V); atol=1e-10)
    end
end

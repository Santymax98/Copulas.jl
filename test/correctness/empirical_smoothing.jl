@testset "Empirical smooth copulas" begin
    U = [
        0.12 0.31 0.54 0.73 0.89 0.42
        0.81 0.22 0.63 0.47 0.15 0.68
    ]

    @testset "Binomial smoother calibration" begin
        S = Copulas.BinomialSmoother()

        for n in (2, 5, 10, 30), u in (0.0, 0.1, 0.37, 0.8, 1.0)
            value = sum(Copulas.κ(S, n, r, u) for r in 1:n) / n
            @test value ≈ u atol=1e-12
        end
    end

    @testset "Binomial smoother recovers BetaCopula" begin
        Cs = EmpiricalSmoothCopula(U, Copulas.BinomialSmoother())
        Cb = BetaCopula(U)

        for u1 in (0.05, 0.2, 0.5, 0.8, 0.97),
            u2 in (0.05, 0.3, 0.6, 0.9, 0.97)

            u = [u1, u2]

            @test cdf(Cs, u) ≈ cdf(Cb, u) atol=1e-12
            @test logpdf(Cs, u) ≈ logpdf(Cb, u) atol=1e-12
        end

        for u in (0.0, 0.1, 0.37, 0.8, 1.0)
            @test cdf(Cs, [u, 1.0]) ≈ u atol=1e-12
            @test cdf(Cs, [1.0, u]) ≈ u atol=1e-12
        end
    end

    @testset "Beta-binomial smoother calibration" begin
        for n in (5, 10, 30), ρ in (2.0, 4.0)
            ρ < n || continue

            S = Copulas.BetaBinomialSmoother(ρ)

            for u in (0.0, 0.1, 0.37, 0.8, 1.0)
                value = sum(Copulas.κ(S, n, r, u) for r in 1:n) / n
                @test value ≈ u atol=1e-12
            end
        end
    end

    @testset "Beta-binomial empirical smooth copula" begin
        C = EmpiricalSmoothCopula(U, Copulas.BetaBinomialSmoother(4.0))

        for u in (0.0, 0.1, 0.37, 0.8, 1.0)
            @test cdf(C, [u, 1.0]) ≈ u atol=1e-12
            @test cdf(C, [1.0, u]) ≈ u atol=1e-12
        end

        @test_throws DomainError EmpiricalSmoothCopula(U,Copulas.BetaBinomialSmoother(6.0),)

        Utest = [
            0.12 0.31 0.54 0.73 0.89
            0.81 0.22 0.63 0.47 0.15
        ]

        R = rosenblatt(C, Utest)
        @test inverse_rosenblatt(C, R) ≈ Utest atol=1e-7
    end

    @testset "Subsetting" begin
        U3 = [
            0.12 0.31 0.54 0.73 0.89 0.42
            0.81 0.22 0.63 0.47 0.15 0.68
            0.34 0.91 0.18 0.57 0.76 0.45
        ]

        C3 = EmpiricalSmoothCopula(U3, Copulas.BinomialSmoother())
        C31 = subsetdims(C3, (3, 1))

        @test C31 isa EmpiricalSmoothCopula

        for u1 in (0.1, 0.4, 0.8), u3 in (0.2, 0.6, 0.9)
            @test cdf(C31, [u3, u1]) ≈ cdf(C3, [u1, 1.0, u3]) atol=1e-12
        end
    end

    @testset "Dependent smoothing" begin
        C = EmpiricalSmoothCopula(U, Copulas.BetaBinomialSmoother(4.0); survival_copula=GaussianCopula(2, 0.6),)

        for u in (0.0, 0.1, 0.37, 0.8, 1.0)
            @test cdf(C, [u, 1.0]) ≈ u atol=1e-12
            @test cdf(C, [1.0, u]) ≈ u atol=1e-12
        end
    end
end
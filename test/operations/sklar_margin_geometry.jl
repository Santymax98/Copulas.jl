
# Internal parameter-geometry contract for joint Sklar MLE.
# Each supported marginal must round-trip through unconstrained
# coordinates and permit likelihood differentiation.

@testset "Sklar marginal parameter geometry" begin

    cases = (
        (Normal, Normal(0.5, 1.3)),
        (Exponential, Exponential(2.0)),
        (Gamma, Gamma(2.5, 1.4)),
    )

    for (Family, original) in cases
        @testset "$Family" begin

            # Natural parameters -> free coordinates
            z = Copulas._sklar_margin_coordinates(original)

            @test length(z) == length(params(original))
            @test all(isfinite, z)

            # Free coordinates -> natural parameters
            recovered = Copulas._sklar_margin_from_coordinates(Family, z)

            @test recovered isa Family

            @test all(isapprox(a, b; rtol=1e-12, atol=1e-12) for (a, b) in zip(params(recovered), params(original)))

            # Differentiation of the marginal log-likelihood
            objective(v) = begin
                margin = Copulas._sklar_margin_from_coordinates(Family, v)
                return sum(logpdf(margin, x) for x in (1.1, 1.7))
            end

            gradient = ForwardDiff.gradient(objective, z)

            @test isfinite(objective(z))
            @test length(gradient) == length(z)
            @test all(isfinite, gradient)
        end
    end

    # Unsupported families must fail explicitly.
    @test_throws ArgumentError Copulas._sklar_margin_schema(Poisson)
end

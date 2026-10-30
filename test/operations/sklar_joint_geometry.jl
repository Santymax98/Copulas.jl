
# Joint parameter geometry for Sklar MLE.
# Tests round-trip reconstruction, coordinate dimensions,
# automatic differentiation, and finite-difference gradients.

@testset "Joint Sklar parameter geometry" begin

    cases = (
        (
            SklarDist(ClaytonCopula{2}(1.5), (Normal(0.3, 1.2), Exponential(1.6)),),
            [-0.4 0.6 1.5; 0.1 0.8 2.0], 4,),
        (
            SklarDist(IndependentCopula{2}(), (Normal(0.3, 1.2), Exponential(1.6)),),
            [-0.4 0.6 1.5; 0.1 0.8 2.0], 3,),
        (
            SklarDist(ClaytonCopula{3}(1.2), (Normal(), Exponential(), Gamma(2.5, 1.4)),),
            [
                -0.4 0.6 1.5
                 0.1 0.8 2.0
                 0.5 1.2 3.0
            ], 6,
        ),
    )

    for (S, X, expected_dimension) in cases
        @testset "$(typeof(S.C)) / $(length(S)) dimensions" begin

            chart = Copulas._sklar_joint_chart(S)
            z = Copulas._sklar_joint_coordinates(chart)

            @test length(z) == expected_dimension
            @test length(chart.ranges) == length(S.m) + 1
            @test all(isfinite, z)

            recovered = Copulas._sklar_joint_distribution(chart, z)

            @test recovered isa SklarDist
            @test params(recovered.C) == params(S.C)

            for i in eachindex(S.m)
                @test all(isapprox(a, b; atol=1e-12, rtol=1e-12) for (a, b) in zip(params(recovered.m[i]), params(S.m[i]),))
            end

            objective(v) = -sum(logpdf(Copulas._sklar_joint_distribution(chart, v), X,))
            # The joint objective must be evaluable in ordinary real arithmetic.
            @test isfinite(objective(z))

            # Independent finite-difference gradient for every supported chart.
            h = 1e-5

            numerical_gradient = map(eachindex(z)) do j
                shift = zeros(eltype(z), length(z))
                shift[j] = h

                return (objective(z + shift) -objective(z - shift)) / (2h)
            end

            @test length(numerical_gradient) == expected_dimension
            @test all(isfinite, numerical_gradient)

            # Gamma CDF differentiation with respect to its shape parameter
            # is not currently supported by ForwardDiff/SpecialFunctions.
            # Its numerical gradient is tested above; ForwardDiff is tested
            # where the complete joint CDF path supports Dual numbers.
            if all(m -> !(m isa Gamma), S.m)
                gradient = ForwardDiff.gradient(objective, z)

                @test all(isfinite, gradient)
                @test all(isapprox(a, b; rtol=2e-4, atol=2e-5) for (a, b) in zip(gradient, numerical_gradient))
            end

        end
    end
end

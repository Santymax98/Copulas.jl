struct ExtremeDist{C} <: Distributions.ContinuousUnivariateDistribution
    G::C
    function ExtremeDist(G)
        return new{typeof(G)}(G)
    end
end

function Distributions.cdf(d::ExtremeDist, z)
    if z < 0
        return 0.0
    elseif z > 1
        return 1.0
    else
        copula = d.G
        return z + z * (1 - z) * (d𝘈(copula, z) / 𝘈(copula, z))
    end
end

function _pdf(d::ExtremeDist, z)
    if z < 0 || z > 1
        return 0.0
    else
        copula = d.G
        A = 𝘈(copula, z)
        A_prime = d𝘈(copula, z)
        A_double_prime = d²𝘈(copula, z)
        return 1 + (1 - 2z) * A_prime / A + z * (1 - z) * (A_double_prime * A - A_prime^2) / A^2
    end
end

function Distributions.quantile(d::ExtremeDist, p)
    if p < 0 || p > 1
        throw(ArgumentError("p must be between 0 and 1"))
    end
    cdf_func(x) = Distributions.cdf(d, x) - p
    return Roots.find_zero(cdf_func, (eps(), 1-eps()), Roots.Brent())
end

# Generate random samples from the radial distribution using the quantile function
function Distributions.rand(rng::Distributions.AbstractRNG, d::ExtremeDist)
    u = rand(rng, Distributions.Uniform(0,1))
    return Distributions.quantile(d, u)
end
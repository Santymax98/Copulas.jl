
# Joint maximum-likelihood optimization for Sklar distributions.

"""
    _fit_sklar_joint_mle(S0, X; weights=nothing)

Maximize the complete Sklar log-likelihood, starting from S0.

Uses ForwardDiff when the joint likelihood supports dual numbers.
Gamma margins currently require finite-difference gradients.
"""
function _fit_sklar_joint_mle(S0::SklarDist, X::AbstractMatrix; weights=nothing)
    chart = _sklar_joint_chart(S0)
    z0 = _sklar_joint_coordinates(chart)
    Xfit, wfit = _weighted_sample(X, weights)

    objective(z) = -_weighted_loglikelihood(_sklar_joint_distribution(chart, z), Xfit, wfit)

    initial = objective(z0)
    isfinite(initial) || throw(ArgumentError("JMLE requires a finite initial log-likelihood"))
    isempty(z0) && return S0

    finite_diff = any(m -> m isa Distributions.Gamma, S0.m)

    result = if finite_diff
        Optim.optimize(objective, z0, Optim.LBFGS())
    else
        Optim.optimize(objective, z0, Optim.LBFGS(); autodiff=ADTypes.AutoForwardDiff())
    end

    zhat = Optim.minimizer(result)
    final = objective(zhat)

    all(isfinite, zhat) && isfinite(final) ||
        throw(ErrorException("JMLE produced non-finite parameters or log-likelihood"))

    Optim.converged(result) ||
        throw(ErrorException("JMLE did not converge after $(Optim.iterations(result)) iterations"))

    tolerance = 1e-8 * max(1.0, abs(initial))
    final <= initial + tolerance ||
        throw(ErrorException("JMLE decreased the log-likelihood relative to its initial fit"))

    return _sklar_joint_distribution(chart, zhat)
end

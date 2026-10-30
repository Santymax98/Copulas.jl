
# Joint unconstrained coordinates for Sklar distributions.
#
# The copula uses its existing Paramorph geometry.
# Margins use the explicitly supported Sklar marginal schemas.

struct _SklarJointChart{S,R}
    prototype::S
    ranges::R
end

_sklar_copula_dimension(C::Copula) = _parameter_dimension(C)
_sklar_copula_dimension(::IndependentCopula) = 0
_sklar_copula_coordinates(C::Copula) = _parameter_coordinates(C)
_sklar_copula_coordinates(::IndependentCopula) = Float64[]
_sklar_copula_reconstruct(C::Copula, z) = _from_parameter_coordinates(C, z)

function _sklar_copula_reconstruct(C::IndependentCopula, z)
    isempty(z) || throw(DimensionMismatch("IndependentCopula has no free parameters",))
    return C
end

# Recover the family constructor without fixing its original numeric type.
# This permits reconstruction with ForwardDiff.Dual parameters.
_sklar_margin_family(m::D) where {D} = Base.typename(D).wrapper

"""
    _sklar_joint_chart(S::SklarDist)

Build an internal parameter chart for a Sklar distribution.

The first coordinate block belongs to the copula, followed by one
block per marginal distribution. Empty blocks are permitted.
"""
function _sklar_joint_chart(S::SklarDist)
    widths = (_sklar_copula_dimension(S.C), map(m -> length(_sklar_margin_coordinates(m)), S.m)...,)
    endpoints = cumsum(collect(widths))
    ranges = ntuple(length(widths)) do i
        (endpoints[i] - widths[i] + 1):endpoints[i]
    end

    return _SklarJointChart(S, ranges)
end

"""
    _sklar_joint_coordinates(chart::_SklarJointChart)

Return the concatenated unconstrained copula and marginal coordinates.
"""
function _sklar_joint_coordinates(chart::_SklarJointChart)
    S = chart.prototype
    copula_z = _sklar_copula_coordinates(S.C)
    margin_z = map(_sklar_margin_coordinates, S.m)

    return vcat(copula_z, margin_z...)
end

"""
    _sklar_joint_distribution(chart::_SklarJointChart, z)

Reconstruct a Sklar distribution from a joint coordinate vector.

The reconstruction preserves automatic-differentiation numeric types.
"""
function _sklar_joint_distribution(chart::_SklarJointChart, z::AbstractVector{<:Real},)
    expected = last(chart.ranges[end])
    length(z) == expected || throw(DimensionMismatch("joint Sklar chart expects $expected coordinates; " * "received $(length(z))",))
    S = chart.prototype
    C = _sklar_copula_reconstruct(S.C, view(z, chart.ranges[1]),)

    margins = ntuple(length(S.m)) do i
        family = _sklar_margin_family(S.m[i])

        _sklar_margin_from_coordinates(family, view(z, chart.ranges[i + 1]),)
    end

    return SklarDist(C, margins)
end

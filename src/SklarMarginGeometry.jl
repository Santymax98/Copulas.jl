# Internal marginal parameter geometry for joint Sklar MLE.
#
# Each supported distribution family is represented by a transformation
# from unconstrained coordinates to its natural parameters.

"""
    _sklar_margin_schema(MarginType)

Return the unconstrained parameter transformation for a supported
univariate margin. Parameter order follows `Distributions.params`.
"""
function _sklar_margin_schema(::Type{D},) where {D<:Distributions.UnivariateDistribution}
    throw(ArgumentError("joint Sklar MLE has no parameter geometry for margin family $D",))
end

_sklar_margin_schema(::Type{<:Distributions.Normal}) = as((asℝ, asℝ₊))
_sklar_margin_schema(::Type{<:Distributions.Exponential}) = as((asℝ₊,))
_sklar_margin_schema(::Type{<:Distributions.Gamma}) = as((asℝ₊, asℝ₊))

"""
    _sklar_margin_coordinates(margin)

Map natural marginal parameters to unconstrained coordinates.
"""
function _sklar_margin_coordinates(margin::Distributions.UnivariateDistribution,)
    schema = _sklar_margin_schema(typeof(margin))
    return Paramorph.TransformVariables.inverse(schema, Distributions.params(margin),)
end

"""
    _sklar_margin_from_coordinates(MarginType, coordinates)

Construct a margin from unconstrained coordinates, preserving
automatic-differentiation numeric types.
"""
function _sklar_margin_from_coordinates(::Type{D}, coordinates::AbstractVector{<:Real},) where {D<:Distributions.UnivariateDistribution}
    schema = _sklar_margin_schema(D)
    parameters = Paramorph.TransformVariables.transform(schema, coordinates,)
    return D(parameters...)
end

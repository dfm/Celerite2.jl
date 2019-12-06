import Base.+

function get_celerite_matrices(
  ar::AbstractArray{T, 1},
  cr::AbstractArray{T, 1},
  ac::AbstractArray{T, 1},
  bc::AbstractArray{T, 1},
  cc::AbstractArray{T, 1},
  dc::AbstractArray{T, 1},
  x::AbstractArray{T, 1},
  diag::AbstractArray{T, 1}
) where T
  N = size(x, 1)
  Jr = size(ar, 1)
  Jc = size(ac, 1)

  a = diag .+ (sum(ar) + sum(ac))
  dx = x[2:N] - x[1:N-1]

  arg = x * transpose(dc)
  ca = cos.(arg)
  sa = sin.(arg)

  U = cat(
    ones(N) * transpose(ar),
    ca .* transpose(ac) + sa .* transpose(bc),
    sa .* transpose(ac) - ca .* transpose(bc),
    dims=2
  )
  V = cat(ones(N, Jr), ca, sa, dims=2)
  Pc = exp.(-dx * transpose(cc))
  P = cat(exp.(-dx * transpose(cr)), Pc, Pc, dims=2)

  return a, U, V, P
end

abstract type Term end

function get_terms(term::Term)
  return tuple(term)
end

function get_coefficients(term::Term)
  return zeros(0), zeros(0), zeros(0), zeros(0), zeros(0), zeros(0)
end

function get_celerite_matrices(term::Term, x, diag)
  return get_celerite_matrices(get_coefficients(term)..., x, diag)
end

struct TermSum <: Term
  terms
end

function get_terms(term::TermSum)
  return term.terms
end

function get_coefficients(term_sum::TermSum)
  ar = zeros(0)
  cr = zeros(0)
  ac = zeros(0)
  bc = zeros(0)
  cc = zeros(0)
  dc = zeros(0)
  for term in term_sum.terms
      coeffs = get_coefficients(term)
      ar = vcat(ar, coeffs[1])
      cr = vcat(cr, coeffs[2])
      ac = vcat(ac, coeffs[3])
      bc = vcat(bc, coeffs[4])
      cc = vcat(cc, coeffs[5])
      dc = vcat(dc, coeffs[6])
  end
  return ar, cr, ac, bc, cc, dc
end

function +(t1::Term, t2::Term)
  return TermSum((get_terms(t1)..., get_terms(t2)...))
end

struct RealTerm <: Term
  a::DenseArray{Float64, 1}
  c::DenseArray{Float64, 1}
  RealTerm(a::Float64, c::Float64) = new(a .+ zeros(1), c .+ zeros(1))
end

function get_coefficients(term::RealTerm)
  return term.a, term.c, zeros(0), zeros(0), zeros(0), zeros(0)
end

struct ComplexTerm <: Term
  a::DenseArray{Float64, 1}
  b::DenseArray{Float64, 1}
  c::DenseArray{Float64, 1}
  d::DenseArray{Float64, 1}
  ComplexTerm(a::Float64, b::Float64, c::Float64, d::Float64) = new(a .+ zeros(1), b .+ zeros(1), c .+ zeros(1), d .+ zeros(1))
end

function get_coefficients(term::ComplexTerm)
  return zeros(0), zeros(0), term.a, term.b, term.c, term.d
end

struct SHOTerm <: Term
  S0::Float64
  w0::Float64
  Q::Float64
end

function get_coefficients(term::SHOTerm)
  eps = 1e-5
  if term.Q < 0.5
    f = sqrt(max(1 - 4 * term.Q^2, eps))
    a = 0.5 * term.S0 * term.w0 * term.Q
    c = 0.5 * term.w0 / term.Q
    return (
      [a * (1 + 1 / f), a * (1 - 1 / f)],
      [c * (1 - f), c * (1 + f)],
      zeros(0), zeros(0), zeros(0), zeros(0)
    )
  end
  f = sqrt(max(4 * term.Q^2 - 1, eps))
  a = term.S0 * term.w0 * term.Q
  c = 0.5 * term.w0 / term.Q
  return zeros(0), zeros(0), [a], [a/f], [c], [c*f]
end

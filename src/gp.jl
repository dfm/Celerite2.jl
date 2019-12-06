using Celerite2

function logp(term::Celerite2.Term, x, y, sigma)

  N = size(x, 1)
  y0 = copy(y)
  a, U, V, P = Celerite2.get_matrices(term, x, sigma.^2 .+ zeros(N))
  Celerite2.factor!(U, P, a, V)
  Celerite2.solve!(U, P, a, V, y0)
  return -0.5 * (transpose(y) * y0 + sum(log.(a)) + N * log(2*pi))

end

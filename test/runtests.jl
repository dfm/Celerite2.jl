using Test
using Random
using LinearAlgebra

using Celerite2

function get_term()
  term = Celerite2.SHOTerm(1.5, 0.8, 2.3)
  term += Celerite2.SHOTerm(1.5, 0.8, 0.3)
  return term
end

function get_coeffs()
  return Celerite2.get_coefficients(get_term())
end

function get_matrices()
  N, x, diag, Z = get_data()
  return Celerite2.get_matrices(get_term(), x, diag)..., Z
end

function get_data()
  Random.seed!(42)
  N = 50
  x = 100 * sort(Random.rand(N))
  diag = 1 .+ Random.rand(N)
  Z = Random.randn(50, 2)
  return N, x, diag, Z
end

function test_to_dense()
  N, x, diag, Z = get_data()
  ar, cr, ac, bc, cc, dc = get_coeffs()

  a, U, V, P, Z = get_matrices()
  K = Celerite2.to_dense(a, U, V, P)

  # Brute force
  tau = abs.(x .- transpose(x))
  K0 = zeros(N, N)
  for j in 1:size(ar, 1)
    K0 = K0 + ar[j] .* exp.(-cr[j] .* tau)
  end
  for j in 1:size(ac, 1)
    K0 = K0 + (ac[j] .* cos.(dc[j] .* tau) + bc[j] .* sin.(dc[j] .* tau)) .* exp.(-cc[j] .* tau)
  end
  K0 += Diagonal(diag)

  return maximum(abs.(K - K0)) < 1e-12
end

function test_matmul()
  a, U, V, P, Z = get_matrices()
  K = Celerite2.to_dense(a, U, V, P)

  Y0 = K * Z
  Y = Celerite2.matmul(a, U, V, P, Z)
  return maximum(abs.(Y - Y0)) < 1e-12
end

function test_factor()
  a, U, V, P, Z = get_matrices()
  K = Celerite2.to_dense(a, U, V, P)

  Celerite2.factor!(U, P, a, V)
  L0 = cholesky(K).L
  L = LowerTriangular(Celerite2.to_dense(ones(size(a, 1)), U, V, P))

  return (
    maximum(abs.(L * Diagonal(a) * transpose(L) - K)) < 1e-12 &&
    maximum(abs.(L * Diagonal(sqrt.(a)) - L0)) < 1e-12
  )
end

function test_dot_tril()
  a, U, V, P, Z = get_matrices()
  K = Celerite2.to_dense(a, U, V, P)

  Celerite2.factor!(U, P, a, V)
  L = cholesky(K).L

  Y0 = L * Z
  Celerite2.dot_tril!(U, P, a, V, Z)
  return maximum(abs.(Z - Y0)) < 1e-12
end

function test_solve()
  a, U, V, P, Z = get_matrices()
  K = Celerite2.to_dense(a, U, V, P)

  Celerite2.factor!(U, P, a, V)
  L = cholesky(K).L

  Y0 = K \ Z
  Celerite2.solve!(U, P, a, V, Z)
  return maximum(abs.(Z - Y0)) < 1e-12
end

@testset "core" begin

  @test test_to_dense()
  @test test_matmul()
  @test test_factor()
  @test test_dot_tril()
  @test test_solve()

end

function test_logp()
  N, x, diag, Z = get_data()
  y = Z[:, 1]
  a, U, V, P, Z = get_matrices()
  K = Celerite2.to_dense(a, U, V, P)

  term = get_term()
  logp = Celerite2.logp(term, x, y, sqrt.(diag))

  chol = cholesky(K)
  logp0 = -0.5 * transpose(y) * (chol \ y)
  logp0 -= 0.5 * logdet(chol)
  logp0 -= 0.5 * N * log(2 * pi)

  return abs(logp - logp0) < 1e-12
end

@testset "gp" begin

  @test test_logp()

end
using LinearAlgebra

function to_dense(
  a::AbstractArray{T, 1},
  U::AbstractArray{T, 2},
  V::AbstractArray{T, 2},
  P::AbstractArray{T, 2}
) where T
  N, J = size(U)
  K::DenseArray{T, 2} = zeros(N, N)

  p = ones(J)
  @inbounds for m in 1:N
    vm = V[m, :]
    fill!(p, 1.0)
    K[m, m] = a[m]

    @inbounds for n in m+1:N
      p = p .* P[n - 1, :]
      un = U[n, :]
      K[n, m] = sum(un .* vm .* p)
      K[m, n] = K[n, m]
    end
  end

  K
end

function matmul(
  a::AbstractArray{T, 1},
  U::AbstractArray{T, 2},
  V::AbstractArray{T, 2},
  P::AbstractArray{T, 2},
  Z::AbstractArray{T}
) where T

  N, J = size(U)
  nrhs = size(Z, 2)

  F = zeros(J, nrhs)
  Y = zeros(N, nrhs)
  Y[N, :] = a[N] .* Z[N, :]
  @inbounds for n in N-1:-1:1
    F = P[n, :] .* (F .+ U[n + 1, :] * transpose(Z[n + 1, :]))
    Y[n, :] = a[n] .* Z[n, :] + transpose(F) * V[n, :]
  end

  fill!(F, 0)
  @inbounds for n in 2:N
    F = P[n - 1, :] .* (F .+ V[n - 1, :] * transpose(Z[n - 1, :]))
    Y[n, :] .+= transpose(F) * U[n, :]
  end

  Y

end

function factor!(
  U::AbstractArray{T, 2},
  P::AbstractArray{T, 2},
  d::AbstractArray{T, 1},
  W::AbstractArray{T, 2}
) where T

  N, J = size(U)
  Sn = zeros(J, J)

  W[1, :] ./= d[1]

  @inbounds for n in 2:N
    Sn += d[n - 1] .* W[n - 1, :] * transpose(W[n - 1, :])
    Sn = P[n - 1, :] .* Sn .* transpose(P[n - 1, :])
    tmp = transpose(U[n, :]) * Sn
    d[n] -= tmp * U[n, :]
    if (d[n] <= 0)
      throw("matrix not positive definite")
    end
    W[n, :] -= transpose(tmp)
    W[n, :] ./= d[n]
  end

end

function solve!(
  U::AbstractArray{T, 2},
  P::AbstractArray{T, 2},
  d::AbstractArray{T, 1},
  W::AbstractArray{T, 2},
  Z::AbstractArray{T}
) where T

  N, J = size(U)
  nrhs = size(Z, 2)
  Fn = zeros(J, nrhs)

  @inbounds for n in 2:N
    Fn = P[n - 1, :] .* (Fn + W[n - 1, :] * transpose(Z[n - 1, :]))
    Z[n, :] -= transpose(Fn) * U[n, :]
  end

  Z ./= d

  fill!(Fn, 0)
  @inbounds for n in N-1:-1:1
    Fn = P[n, :] .* (Fn + U[n + 1, :] * transpose(Z[n + 1, :]))
    Z[n, :] -= transpose(Fn) * W[n, :]
  end

end

function dot_tril!(
  U::AbstractArray{T, 2},
  P::AbstractArray{T, 2},
  d::AbstractArray{T, 1},
  W::AbstractArray{T, 2},
  Z::AbstractArray{T}
) where T

  N, J = size(U)
  nrhs = size(Z, 2)
  Fn = zeros(J, nrhs)
  sqrtd = sqrt.(d)

  Z[1, :] .*= sqrtd[1]
  tmp = Z[1, :]

  @inbounds for n in 2:N
    Fn = P[n - 1, :] .* (Fn + W[n - 1, :] * transpose(tmp))
    tmp = sqrtd[n] .* Z[n, :]
    Z[n, :] = tmp + transpose(Fn) * U[n, :]
  end

end

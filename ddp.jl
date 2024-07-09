using ForwardDiff
using Plots 
using Random
using Tullio # for tensor multiplication

function DDP_iteration(X, U, loss, loss_final, state_eq; α = 1)
    n, N = size(X)
    m = size(U, 1)

    V = zeros(N)
    Vx = zeros(n, N)
    Vxx = zeros(n, n, N)

    V[end] = loss_final(X[:,end])
    Vx[:, end] .= ForwardDiff.gradient(loss_final, X[:, end])
    Vxx[:, :, end] .= ForwardDiff.hessian(loss_final, X[:, end])

    # backward pass
    k_gain = zeros(N-1,m,m)
    K = zeros(N-1,m, n)
    for k = N-1:-1:1
        fx = ForwardDiff.jacobian(x -> state_eq(x, U[:,k]), X[:, k])
        fu = ForwardDiff.jacobian(u -> state_eq(X[:, k], u), U[:,k])
        lx = ForwardDiff.gradient(x -> loss(x, U[:,k]), X[:, k])
        lu = ForwardDiff.gradient(u -> loss(X[:, k], u), U[:,k])
        lxx = ForwardDiff.jacobian(x -> ForwardDiff.gradient(x -> loss(x, U[:,k]), x), X[:,k])
        luu = ForwardDiff.hessian(u -> loss(X[:, k], u), U[:,k])
        lux = ForwardDiff.jacobian(u -> ForwardDiff.gradient(x -> loss(x, u),X[:,k]),U[:,k])

        # ForwardDiff returns the 3d hessian with stacked last dimension, so have to reshape
        fxx = reshape(
            ForwardDiff.jacobian(x -> ForwardDiff.jacobian(x -> state_eq(x,U[:,k]), x), X[:,k]),
            n,n,n
        )
        fuu = reshape(
            ForwardDiff.jacobian(u -> ForwardDiff.jacobian(u -> state_eq(X[:,k], u), u), U[:,k]),
            n,m,m
        )
        fux = reshape(
            ForwardDiff.jacobian(u -> ForwardDiff.jacobian(x -> state_eq(x,u), X[:,k]), U[:,k]),
            n,n,m
        )
        @tullio term_xxx[j,l] := Vx[i,k+1]*fxx[i,j,l]
        @tullio term_xux[j,l] := Vx[i,k+1]*fux[i,j,l]
        @tullio term_xuu[j,l] := Vx[i,k+1]*fuu[i,j,l]

        Qx = lx + fx'*Vx[:, k+1]
        Qu = lu + fu'*Vx[:, k+1]
        Qxx = lxx + fx'*Vxx[:,:,k+1]*fx + term_xxx
        Qux = lux + (fu'*Vxx[:,:,k+1]*fx)' + term_xux
        Quu = luu + (fu'*Vxx[:,:,k+1]*fu) + term_xuu

        if size(Quu) == (1,1)
            Quu = Quu[1]
        end
        k_gain[k,:,:] = -Quu \ Qu
        K[k,:,:] = (-Quu \ Qux)'

        Vx[:,k] = Qx - K[k,:,:]'*Quu*k_gain[k,:,:]
        Vxx[:,:,k] = Qxx - K[k,:,:]'*Quu*K[k,:,:]
        V[k] = V[k+1] + (0.5*k_gain[k,:,:]'*Quu*k_gain[k,:,:])[1]
        
    end

    # forward pass
    X_new = zeros(n, N)
    X_new[:,1] = x0
    U_new = zeros(m,N-1)
    for k in 1:N-1
        U_new[:,k] = U[:,k] + α * k_gain[k,:,:] + K[k,:,:]*(X_new[:,k] - X[:,k])
        X_new[:,k+1] = state_eq(X_new[:,k], U_new[:,k])
    end
    return X_new, U_new, V
end

function loss(x::AbstractVector, U)
    10*(x[1] - 5)^2 + x[2]^2 + 10*U[1]^2
end

function loss_final(x::Vector)
    (x[1] - 5)^2 + x[2]^2
end

function state_eq(x::Vector, u)
    T = 0.1
    x_new = [1 T; 0 1]*x + [0.5*T^2; T]*u[1]
end

T_final = 10.
N = 100 # Number of time steps
dt = T_final / N

n = 2   # State dimension
m = 1   # Control dimension


Random.seed!(1234)
U = rand(m,N-1) * 0.3 .- 0.15   # Random control sequence
# U = [0.3*ones(m, N÷2) -0.2*ones(m, N÷2)]

x0 = [0., -0.3] # Initial state
X = zeros(n, N)
X[:,1] = x0
for k in 2:N
    X[:,k] = state_eq(X[:,k-1], U[k-1]) # Obtain initial guess state trajectory
end

Xs = [X]    # Store state trajectories
Us = [U]    # Store control trajectories
Vs = []     # Store value function

iterations = 15
for k in 1:iterations
    Xnew, Unew, V = DDP_iteration(X, U, loss, loss_final, state_eq, α=0.5)  # Perform DDP
    push!(Xs, Xnew) # Store outputs
    push!(Us, Unew)
    push!(Vs, V)
    X = Xnew
    U = Unew
end

## Visualisation
plt_state = plot()
for k in 1:iterations
    plot!(plt_state, Xs[k][1,:], label=false, title="State")
end
display(plt_state)

plt_ctrl = plot()
for k in 1:iterations
    plot!(plt_ctrl, Us[k][1,:], label=false, title="Control")
end
display(plt_ctrl)

plt_V = plot()
for k in 1:iterations
    plot!(plt_V, Vs[k], label=false, title="Value function")
end
display(plt_V)
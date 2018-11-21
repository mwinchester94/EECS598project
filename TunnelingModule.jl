using Flux, Flux.Data.MNIST ## this is the Julia package for deep learning 
using Flux: onehotbatch, argmax, crossentropy, throttle, mse
using Base.Iterators: repeated, partition
using Statistics, LinearAlgebra
using Distributed #for parallel computation

#least squares regression
function lin_reg(A, b)
    n,p = size(A)
    A = hcat(ones(n,1),A)
    x = LinearAlgebra.pinv(A)*b
    return x
end

#center column of data x about mean value
function colcenter(x)
    cent = x .- mean(x)
    return cent
end

#Calculate break:tunnel ratio and release:tunnel ratio
#ratios calculed by taking the l2 norm of each pitch compared to the average pitch at the pitch, tunnel, and base locations (x,z coords) 
#inputs are data X = [ReleasePositionX, ReleasePositionZ, TunnelLocationX, TunnelLocationZ, TrajectoryLocationX, TrajectoryLocationZ]
function calc_ratios(X)
    releasedist = (colcenter(X[:,1]) .^ 2 + colcenter(X[:,2]) .^ 2) .^ 0.5
    tunneldist = (colcenter(X[:,3]) .^ 2 + colcenter(X[:,4]) .^ 2) .^ 0.5
    breakdist = (colcenter(X[:,5]) .^ 2 + colcenter(X[:,6]) .^ 2) .^ 0.5
    
    btratio = breakdist ./ tunneldist
    rtratio = releasedist ./ tunneldist
    return hcat(btratio,rtratio)
end

#calculate ratios of data X, then perform regression of y onto ratios
function lin_reg_ratios(X,y)
    return lin_reg(calc_ratios(X),y)
end

#perform linear regression with ratios, then calculate mse with ytest
function lin_reg_mse(Xtrain,ytrain,Xtest,ytest)
    n,p = size(Xtest)
    
    coef = lin_reg_ratios(Xtrain,ytrain)
    ypred = hcat(ones(n,1),calc_ratios(Xtest)) * coef
    return coef, mse(ypred,ytest)
end

#train a neural network on set of features in X (transformed or otherwise) and calculate mse
function nn_full(Xtrain,ytrain,Xtest,ytest)
    p = size(Xtrain,2)
    
    loss_fn = mse
    n = 8
    m = Chain(Dense(p, n),Dense(n,2*n,relu),Dense(2*n,n,sigmoid),Dense(n,1))  
    loss(x, y) = loss_fn(m(x), y) 
    evalcb = () -> @show([loss(Xtrain',ytrain')])
    dataset = Base.Iterators.repeated((Xtrain', ytrain'), 500)
    opt = ADAM(params(m)) 
    Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 0.01))
    
    return params(m), mse(m(Xtest'),ytest'), m(Xtest') - ytest' 
end

#k nearest neighbors functions

#calculate l2 norm squared of two vectors
@everywhere function euclidean_distance(a, b)
 distance = 0.0 
 for index in 1:size(a, 1) 
  distance += (a[index]-b[index]) * (a[index]-b[index])
 end
 return distance
end

#returns estimate yest from average of k nearest neighbors in Xtrain for a single test point xtest 
@everywhere function nearest_neighbor(Xtrain,ytrain,xtest,k=3)
    n,p = size(Xtrain)
    dist = zeros(n)
    for i = 1:n
        dist[i] = euclidean_distance(Xtrain[i],xtest')
    end
    
    sortedNeighbors = sortperm(dist)
    yest = 0
    for j = 1:k
        yest = yest + ytrain[sortedNeighbors[j]]
    end
    return yest/k
end

#compute knn estimate for each sample in Xtest and calculate mse
function knn(Xtrain,Ytrain,Xtest,Ytest,k=3)
    nsamp = length(Ytest)
    Yest = zeros(nsamp,1)
    #for i = 1:nsamp
    #    Yest[i] = nearest_neighbor(Xtrain,Ytrain,Xtest[i,:],k)
    #end
    Yest = @distributed (vcat) for i = 1:nsamp
        nearest_neighbor(Xtrain,Ytrain,Xtest[i,:],k)
    end
    return Yest, mse(Yest,Ytest)
end
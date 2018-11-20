using Flux, Flux.Data.MNIST ## this is the Julia package for deep learning 
using Flux: onehotbatch, argmax, crossentropy, throttle, mse
using Base.Iterators: repeated, partition
using Statistics

#least squares regression
function lin_reg(A, b)
    n,p = size(A)
    A = hcat(ones(n,1),A)
    x = pinv(A)*b
    return x
end

#center column of data x about mean value
function colcenter(x)
    cent = x - mean(x)
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
    return mse(ypred,ytest)
end

#train a neural network on set of features in X (transformed or otherwise) and calculate mse
function nn_full(Xtrain,ytrain,Xtest,ytest)
    p = size(Xtrain,2)
    
    loss_fn = mse
    n = 10
    m = Chain(Dense(p, n),Dense(n,n,relu),Dense(n,1))  
    loss(x, y) = loss_fn(m(x), y) 
    evalcb = () -> @show([loss(Xtrain',ytrain')])
    dataset = Base.Iterators.repeated((Xtrain', ytrain'), 500)
    opt = ADAM(params(m)) 
    Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 0.01))
    
    return params(m), mse(m(Xtest'),ytest'), m(Xtest') - ytest' 
end
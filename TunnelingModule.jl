using GLM, Plots, JLD
using Flux, Flux.Data.MNIST ## this is the Julia package for deep learning 
using Flux: onehotbatch, argmax, crossentropy, throttle, mse
using Base.Iterators: repeated, partition
using Statistics: mean
using LinearAlgebra: pinv


##preprocessing

function connect_db()
    connection_dir = "./postgresql-files/pitch-tunneling"

    user, pw = open("pw_pitch_tunneling.txt") do file
        readlines(file)
    end

    username = user
    password = pw

    return connect_pitch_tunneling(connection_dir, username, password)
end

#output of query to usable data format; returns tuple of (X,Y)
function dataframe2matrix(data)
    X = convert(Array,data[1:end-1])
    Y = parse.(Float64,data[end])
    return (X, Y)
end

#get data from database; returns tuple of (X,Y)
function get_data(train=true; islhp=false, islhb=false, fast=true)
    #make the query based on the selections; all but first keyword arguments
    query = "SELECT releasepositionx, releasepositionz, tunnellocationx, tunnellocationz, trajectorylocationx, trajectorylocationz, runvalue\nFROM "

    if train
        query *= "pitchtunnelingtrain\nWHERE "
    else
        query *= "pitchtunnelingtest\nWHERE "
    end

    query *= "islhp = '"*string(Int(islhp))*"' AND islhb = '"*string(Int(islhb))*"' AND "

    if fast
        query *= "(pitchtype = 'FF' OR pitchtype = 'FT' OR pitchtype = 'SI') AND runvalue IS NOT NULL AND runvalue != 'NA';"
    else
        query *= "(pitchtype != 'FF' AND pitchtype != 'FT' AND pitchtype != 'SI') AND runvalue IS NOT NULL AND runvalue != 'NA';"
    end

    #connect to database and send query
    conn = connect_db()
    data = stream_to_dataframe(conn, query)
    close(conn)
    return dataframe2matrix(data) #returns tuple of (X,Y)
end


##fitting/training algos

#calculate l2 norm squared of a vector
function euclidean_distance(a, b)
    distance = 0.0 
    for index in 1:size(a, 1) 
        distance += (a[index]-b[index]) * (a[index]-b[index])
    end
    return distance
end

function taxicab_distance(a,b)
    distance = 0.0
    for index in 1:size(a,1)
        distance += abs(a[index]-b[index])
    end
    return distance
end

#center columns of matrix X about mean value of columns
function colcenter(X)
    cent = X - ones(size(X,1))*mean(X,dims=1)
    return cent
end

#calculate distances from mean
function calc_distance(X)
    X_m = colcenter(X)
    releasedist = sqrt.(X_m[:,1].^2 + X_m[:,2].^2)
    tunneldist = sqrt.(X_m[:,3].^2 + X_m[:,4].^2)
    breakdist = sqrt.(X_m[:,5].^2 + X_m[:,6].^2)
    
    return hcat(releasedist,tunneldist,breakdist)
end

#Calculate break:tunnel ratio and release:tunnel ratio
#ratios calculed by taking the l2 norm of each pitch compared to the average pitch at the pitch, tunnel, and base locations (x,z coords) 
#inputs are data tuple, X = [ReleasePositionX, ReleasePositionZ, TunnelLocationX, TunnelLocationZ, TrajectoryLocationX, TrajectoryLocationZ] and Y
function calc_ratios(data)
    dists = calc_distance(data[1])
    
    btratio = dists[:,3] ./ dists[:,2]
    rtratio = dists[:,1] ./ dists[:,2]
    
    #triple = dists[:,1].*dists[:,2]./dists[:,3]
    #triple = reshape(triple, (size(triple,1),1))
    return (hcat(btratio,rtratio), data[2])
end

#least squares regression
function lin_reg(A, b)
    n,p = size(A)
    A = hcat(ones(n,1),A)
    x = pinv(A)*b
    return x
end

#prediction from linear regression coefficients
function lin_reg_pred(Xtrain,Ytrain,Xtest)
    n,p = size(Xtest)
    ypred = hcat(ones(n,1),Xtest) * lin_reg(Xtrain,Ytrain)
    return ypred
end

#train a neural network on set of features in X (transformed or otherwise) and return model
function nn_full(Xtrain,ytrain)
    p = size(Xtrain,2)
    
    loss_fn = mse
    n = 8
    m = Chain(Dense(p, n),Dense(n,n,sigmoid),Dense(n,1))  
    loss(x, y) = loss_fn(m(x), y) 
    evalcb = () -> @show(loss(Xtrain',ytrain'))
    dataset = Base.Iterators.repeated((Xtrain', ytrain'), 500)
    opt = ADAM(params(m)) 
    #Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 0.01))
    Flux.train!(loss, dataset, opt)
    evalcb
    
    return m 
end

function nn_alt(Xtrain,ytrain)
    p = size(Xtrain,2)
    
    loss_fn = mse
    n = 8
    m = Chain(Dense(p, n),Dense(n,n,sigmoid),Dense(n,1))  
    loss(x, y) = loss_fn(m(x), y) 
    evalcb = () -> @show([loss(Xtrain',ytrain')])
    opt = ADAM(params(m))
    for idx = 1 : 500
        dataset = Base.Iterators.repeated((Xtrain', ytrain'), 1)
        Flux.train!(loss, dataset, opt)
    end
    evalcb
    
    return m 
end

function nn_pred_orig(m,Xtest)
    return m(Xtest')'
end

function nn_pred_params(p,Xtest)
    return (p[5]*sigmoid.(p[3]*(p[1]*Xtest' .+ p[2]) .+ p[4]) .+ p[6])'
end

#returns estimate ypred from average of k nearest neighbors in Xtrain for a single test point xtest 
#neighbors are weighted by 1/d where d is the distance to the neighbor
function nearest_neighbor(Xtrain,ytrain,xtest,k=3)
    n,p = size(Xtrain)
    dist = zeros(n)
    for i = 1:n
        dist[i] = euclidean_distance(Xtrain[i],xtest')
    end
    
    sortedNeighbors = sortperm(dist)
    ypred = 0
    invdistsum = 0.0 #sum of inverted distances, used for assigning weights
    for j = 1:k
        ypred = ypred + ytrain[sortedNeighbors[j]]/dist[j]
        invdistsum += 1.0/dist[j]
    end
    return ypred/invdistsum
end

#compute knn estimate for each sample in Xtest
function knn_pred(Xtrain,Ytrain,Xtest,k=3)
    nsamp = size(Xtest,1)
    Ypred = zeros(nsamp,1)
    for i = 1:nsamp
        Ypred[i] = nearest_neighbor(Xtrain,Ytrain,Xtest[i,:],k)
    end
    return Ypred
end


##evaluation functions

#mse: function already exists

function calc_acc_dec(pred, Y)
    p = round.(pred, digits=1)
    y = round.(Y, digits=1)
    return 100*sum(p .== y)/length(y)
end

function calc_acc_int(pred, Y)
    p = round.(pred, digits=0)
    y = round.(Y, digits=0)
    return 100*sum(p .== y)/length(y)
end

#function calc_acc_nonzero(pred, Y)
#    #compare all entries in Y that are not rounded to 0
#    p = round.(pred, digits=0)
#    y = round.(Y, digits=0)
#    return sum((y .== p) .* (y .!= 0))/sum(y .!= 0)
#end
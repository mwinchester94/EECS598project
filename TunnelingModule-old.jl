using GLM, Plots
using Flux, Flux.Data.MNIST ## this is the Julia package for deep learning 
using Flux: onehotbatch, argmax, crossentropy, throttle, mse
using Base.Iterators: repeated, partition
using Statistics: mean
using LinearAlgebra: pinv
using Distributed #for parallel computation


##preprocessing

function connect_db()
    connection_dir = "./pitch-tunneling"

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
function get_data(train=true; islhp=false, islhb=false, fast=true, sorted=false)
    #make the query based on the selections; all but first keyword arguments
    query = "SELECT "

    if sorted
        query *= "gamepk, inning, istop, atbatnumber, pitchnumber, "
    end

    query *= "releasepositionx, releasepositionz, tunnellocationx, tunnellocationz, trajectorylocationx, trajectorylocationz, runvalue\nFROM "

    if train
        query *= "pitchtunnelingtrain\nWHERE "
    else
        query *= "pitchtunnelingtest\nWHERE "
    end

    query *= "islhp = '"*string(Int(islhp))*"' AND islhb = '"*string(Int(islhb))*"' AND "

    if sorted
        query *= "runvalue IS NOT NULL AND runvalue != 'NA'\nORDER BY gamepk, inning, istop DESC, atbatnumber, pitchnumber;"
    else
        if fast
            query *= "(pitchtype = 'FF' OR pitchtype = 'FT' OR pitchtype = 'SI') AND runvalue IS NOT NULL AND runvalue != 'NA';"
        else
            query *= "(pitchtype != 'FF' AND pitchtype != 'FT' AND pitchtype != 'SI') AND runvalue IS NOT NULL AND runvalue != 'NA';"
        end
    end

    #connect to database and send query
    conn = connect_db()
    data = stream_to_dataframe(conn, query)
    close(conn)
    return dataframe2matrix(data) #returns tuple of (X,Y)
end


##fitting/training algos

#least squares regression
function lin_reg(A, b)
    n,p = size(A)
    A = hcat(ones(n,1),A)
    x = pinv(A)*b
    return x
end

#center column of data x about mean value
function colcenter(x)
    cent = x .- mean(x)
    return cent
end

#calculate distances from either mean or previous pitch; input tuple (X,Y), output (X_dist, Y) where X_dist is matrix of distances and Y is only changed for sequential case
function calc_distance(data; seq=false)
    X, Y = data[1], data[2]
    if seq
        #data will be starting at col 6, col 5 is pitch number
        releasedist = []
        tunneldist = []
        breakdist = []
        Y_new = []
        for idx in 2:size(X,1)
            if (X[idx,5]-1) == X[idx-1,5]
                push!(releasedist, sqrt((X[idx,6]-X[idx-1,6])^2 + (X[idx,7]-X[idx-1,7])^2))
                push!(tunneldist, sqrt((X[idx,8]-X[idx-1,8])^2 + (X[idx,9]-X[idx-1,9])^2))
                push!(breakdist, sqrt((X[idx,10]-X[idx-1,10])^2 + (X[idx,11]-X[idx-1,11])^2))
                push!(Y_new, Y[idx])
                end
        end
    else
        releasedist = sqrt.(colcenter(X[:,1]) .^ 2 + colcenter(X[:,2]) .^ 2)
        tunneldist = sqrt.(colcenter(X[:,3]) .^ 2 + colcenter(X[:,4]) .^ 2)
        breakdist = sqrt.(colcenter(X[:,5]) .^ 2 + colcenter(X[:,6]) .^ 2)
        Y_new = Y
    end
    
    return (hcat(releasedist,tunneldist,breakdist), Y_new)
end

#Calculate break:tunnel ratio and release:tunnel ratio
#ratios calculed by taking the l2 norm of each pitch compared to the average pitch at the pitch, tunnel, and base locations (x,z coords) 
#inputs are data tuple, X = [ReleasePositionX, ReleasePositionZ, TunnelLocationX, TunnelLocationZ, TrajectoryLocationX, TrajectoryLocationZ] and Y
function calc_ratios(data; seq=false)
    dists, Y = calc_distance(data, seq=seq)
    
    btratio = dists[:,3] ./ dists[:,2]
    rtratio = dists[:,1] ./ dists[:,2]
    
    triple = dists[:,1].*dists[:,2]./dists[:,3]
    return (dists, hcat(btratio,rtratio), reshape(triple, (length(triple),1)), Y)
end

#prediction from linear regression coefficients
function lin_reg_pred(coef,Xtest)
    n,p = size(Xtest)
    ypred = hcat(ones(n,1),calc_ratios(Xtest)) * coef
    return ypred
end

#train a neural network on set of features in X (transformed or otherwise) and return model
function nn_full(Xtrain,ytrain)
    p = size(Xtrain,2)
    
    #loss_fn = mse
    n = 8
    m = Chain(Dense(p, n),Dense(n,n,sigmoid),Dense(n,1))  
    loss(x, y) = loss_acc(m(x), y) 
    evalcb = () -> @show(loss(Xtrain',ytrain'))
    dataset = Base.Iterators.repeated((Xtrain', ytrain'), 500)
    opt = ADAM(params(m)) 
    Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 0.01))
    #Flux.train!(loss, dataset, opt)
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

function nn_pred(m,Xtest)
    return m(Xtest')'
end


##evaluation functions

#mse: function already exists

function calc_acc_dec(pred, Y)
    p = round.(pred, digits=1)
    y = round.(Y, digits=1)
    return sum(p .== y)/length(y)
end

function calc_acc_int(pred, Y)
    p = round.(pred, digits=0)
    y = round.(Y, digits=0)
    return sum(p .== y)/length(y)
end

#1-acc for nn minimization
function loss_acc(x,y)
    return 1-calc_acc_dec(x,y)
end

#fct of how many different from 0 correcly predicted?

# opposite of acc for nn? Y = parse.(Float64,data2[end])';
#@show size(Y)
#p = x -> (x >= 0.1 || x<-0.1)
#@show count(p, Y)


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

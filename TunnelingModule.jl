using GLM, Plots
using Flux, Flux.Data.MNIST ## this is the Julia package for deep learning 
using Flux: onehotbatch, argmax, crossentropy, throttle, mse
using Base.Iterators: repeated, partition
using Statistics, LinearAlgebra


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
    cent = x - mean(x)
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
    return (dists, hcat(btratio,rtratio), triple, Y)
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


##evaluation functions

#mse: function already exists

function calc_accuracy(pred, Y)
    p = round.(pred, digits=1)
    y = round.(Y, digits=1)
    return sum(p .== y)/length(y)
end
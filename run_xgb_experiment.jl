#!/usr/bin/env julia

# -- Import basic libraries for test
using Pkg
using SparseArrays

# -- Auxiliary Functions for test

function getdata(filename::String; rows::Int=-1)::Tuple{SparseMatrixCSC, Vector{Float32}, Vector{Int}}
    """Get feature as sparse matrix, labels and group ids after reading a SVMLight formatted file"""
    I, J, V, ys, qids = Int[], Int[], Float32[], Int[], Int[]
    for (i, rawrow) in enumerate(eachline(filename))
        row = split(rawrow, " ")
        push!(ys, parse(Float32, row[1]))
        push!(qids, parse(Int, last(split(row[2], ":"))))
        for f in row[3:end]
            j, v = split(f, ":")
            push!(I, i)
            push!(J, parse(Int, j))
            push!(V, parse(Float32, v))
        end
        rows > 0 && length(ys) >= rows && break
    end
    sparse(I, J, V), ys, countqids(qids)
end

function countqids(qids::Vector{Int})::Vector{Int}
    """Return a count of qids. Example: qid input '[150, 150, 21, 21, 5]' return '[2, 2, 1]'."""
    last_qid, counts = first(qids) , Int[0]
    for qid in qids
        if last_qid == qid
            counts[end] += 1
        else
            push!(counts, 1)
            last_qid = qid
        end
    end
    counts
end

function setgroup(dmatrix, qids::Vector{Int})
    """Set groups 'qids' to DMatrix"""
    group = convert(Vector{UInt32}, qids)
    group_size = convert(UInt64, size(group, 1))
    XGBoost.XGDMatrixSetUIntInfo(dmatrix.handle, "group", group, group_size)
end

function precision_at_k(y, yhat, qids; k::Int, threshold::Float64=0.0)::Float32
    """Return metric precision at k"""
    score = 0.0
    s = 1
    for qid in qids
        allowed_k = convert(Int, min(qid, k))
        e = s + allowed_k - 1
        hits = sum((yhat[s:e] .> threshold) .== (y[s:e] .> threshold))
        score += hits/allowed_k
        s += qid
    end

    round(score/length(qids), digits=5)
end

function confusion_matrix(ys, yshat, qids; threshold::Float64=0.0)
    """Show confusion matrix"""
    s = 1
    tp, fp, tn, fn = 0, 0, 0, 0
    for qid in qids
        e = s + qid - 1
        for (yhat, y) in zip(yshat[s:e] .> threshold, ys[s:e] .> threshold)
            if yhat == y
                if yhat == 1
                    tp += 1
                else
                    tn += 1
                end
            else
                if yhat == 1
                    fp += 1
                else
                    fn += 1
                end
            end
        end
        s += qid
    end

    precision = round(tp/(tp+fp), digits=5)
    recall = round(tp/(tp+fn), digits=5)
    f1 = round(2 * (precision * recall)/(precision + recall), digits=5)
    accuracy = round((tp+tn)/(tp+fp+tn+fn), digits=5)
    prevalence = round((tp+fn)/(tp+fn+fp+tn), digits=5)
    @info "TP:$tp FP:$fp TN:$fn FN$fn Precision:$precision Recall:$recall F1:$f1 Accuracy:$accuracy Prevalence:$prevalence"
end

function evalmetrics(bst, dtrain, y_train, qids_train, dtest, y_test, qids_test; threshold::Float64=0.0)
    """Evaluate model quality in training and test data using Precision@N metric"""

    yhat_train = XGBoost.predict(bst, dtrain)
    yhat_test  = XGBoost.predict(bst, dtest)

    save_predictions(yhat_train, y_train, "train_prediction_xgb_v$(VERSION).csv")
    save_predictions(yhat_test, y_test, "test_prediction_xgb_v$(VERSION).csv")

    pat5  = precision_at_k(y_train, yhat_train, qids_train, k=5,  threshold=threshold)
    pat10 = precision_at_k(y_train, yhat_train, qids_train, k=10, threshold=threshold)
    pat20 = precision_at_k(y_train, yhat_train, qids_train, k=20, threshold=threshold)
    @info "(1) TRAIN - Precision@N: p@5:$pat5 p@10:$pat10 p@20:$pat20"
    #confusion_matrix(y_train, yhat_train, qids_train; threshold=threshold)

    pat5  = precision_at_k(y_test, yhat_test, qids_test, k=5,  threshold=threshold)
    pat10 = precision_at_k(y_test, yhat_test, qids_test, k=10, threshold=threshold)
    pat20 = precision_at_k(y_test, yhat_test, qids_test, k=20, threshold=threshold)
    @info "(2) TEST - Precision@N: p@5:$pat5 p@10:$pat10 p@20:$pat20"
    #confusion_matrix(y_test, yhat_test, qids_test; threshold=threshold)
end

function ConvertDMatrix(x::SparseMatrixCSC{<:Real,<:Integer}; kw...)
    """Transform sparse matrix 'x' to DMatrix in XGBoost v.2 generation"""
    o = Ref{XGBoost.DMatrixHandle}()
    (colptr, rowval, nzval) = XGBoost._sparse_csc_components(x)
    XGBoost.xgbcall(XGBoost.XGDMatrixCreateFromCSCEx, colptr, rowval, nzval,
                    size(colptr,1), nnz(x), size(x,1), o)
    XGBoost.DMatrix(o[]; kw...)
end

function save_predictions(ypred::Vector, yreal::Vector, filename::String)
    """Save predictions into CSV format with columns 'yreal' and 'ypred'"""
    io = open(filename, "w")
    write(io, "yreal,ypred\n")
    for i=1:length(ypred)
        write(io, "$(yreal[i]),$(ypred[i])" * (i < length(ypred) ? "\n" : ""))
    end
    flush(io)
    close(io)
end
    
# -- Code Execution

# Parameters
VERSION = get(ENV, "VERSION", "1.5.2")
NUM_ITERATIONS = parse(Int, get(ENV, "ITE", "10"))

println("\nXGBoost v1 x v2 Experiment\n")

# Define parameters for XGBoost model
params = Dict(
    "seed" => 1,
    "num_round" => NUM_ITERATIONS,
    "booster" =>"gbtree",
    "objective" => "rank:pairwise",
    "verbosity" => 1,
    "eta" => 0.01,
    "gamma" => 0,
    "max_depth" => 7,
    "min_child_weight" => 1,
    "max_delta_step" => 0,
    "subsample" => 0.9,
    "colsample_bytree" => 0.9,
    "colsample_bylevel" => 0.9,
    "colsample_bynode" => 1.0,
    "lambda" => 1,
    "alpha" => 0,
    "refresh_leaf" => 1,
    "process_type" => "default",
    "tree_method" => "hist",
    "num_parallel_tree" => 1,
    "grow_policy" => "depthwise",
    "max_bin" => 256,
    "predictor" => "auto" )

@info "-- Loading Data..."
x_train, y_train, qids_train = getdata("train.svmlight")
x_test, y_test, qids_test = getdata("test.svmlight")

@info "Train Data X:$(size(x_train)) Y:$(length(y_train)) QIDs:$(length(qids_train))"
@info "Test Data X:$(size(x_test)) Y:$(length(y_test)) QIDs:$(length(qids_test))"
@info "-- ended data loading"; println()

if occursin(r"^1[.]", VERSION) 
    @info "Execute Analysis for XGBoost v$VERSION"
    Pkg.add(Pkg.PackageSpec(name="XGBoost", version=VERSION), io=devnull)
    using XGBoost
    
    dtrain    = XGBoost.makeDMatrix(x_train, y_train)
    dtest     = XGBoost.makeDMatrix(x_test, y_test)
    watchlist = [(dtrain, "train"), (dtest, "eval")]

    setgroup(dtrain, qids_train)
    setgroup(dtest, qids_test)
    
    bst = XGBoost.xgboost(dtrain, NUM_ITERATIONS, metrics=["auc"], watchlist=watchlist, param=params)
    
    println("\nResults Run XGBoost v$VERSION")
    evalmetrics(bst,
                XGBoost.DMatrix(x_train), y_train, qids_train,
                XGBoost.DMatrix(x_test),  y_test,  qids_test,
                threshold=0.0)

    println("\nInvert Test and Train for Evaluation")
    evalmetrics(bst,
                XGBoost.DMatrix(x_test), y_test, qids_test,
                XGBoost.DMatrix(x_train),  y_train,  qids_train,
                threshold=0.0)
    
end

if occursin(r"^2[.]", VERSION) 
    @info "Execute Analysis for XGBoost v$VERSION"
    Pkg.add(Pkg.PackageSpec(name="XGBoost", version=VERSION), io=devnull)
    using XGBoost

    dtrain = ConvertDMatrix(x_train, label=y_train)
    dtest  = ConvertDMatrix(x_test, label=y_test)

    setgroup(dtrain, qids_train)
    setgroup(dtest, qids_test)
    
    params["watchlist"] = Dict("train" => dtrain, "eval" => dtest)
    params["eval_metric"] = "auc"
    kwargs = Dict{Symbol, Any}(Symbol(k) => v for (k,v) in params)
    
    bst = XGBoost.xgboost(dtrain; kwargs...)
    
    println("\nResults Run XGBoost v$VERSION")
    evalmetrics(bst,
                ConvertDMatrix(x_train), y_train, qids_train,
                ConvertDMatrix(x_test),  y_test,  qids_test, threshold=0.0)

    println("\nInvert Test and Train for Evaluation")
    evalmetrics(bst,
                ConvertDMatrix(x_test), y_test, qids_test,
                ConvertDMatrix(x_train),  y_train,  qids_train, threshold=0.0)
end

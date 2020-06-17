using Pkg
using CSV
using DataFrames
using Statistics
# Pkg.add("IJulia")
# Pkg.add("DataFrames")
# Pkg.add("CSV")


#cd("C:/Users/Guenter/Google Drive/medium/jupyR")
cd("C:/Users/groehrich/Google Drive/medium/jupyR")
pwd()

data = CSV.read("./iris_data.txt",header=false)

## no more head
first(data,5)

# renaming column header
rename!(data, Symbol.(["sepal_length","sepal_width","petal_length","petal_width","class"]))

unique(data."class")

## map values to class - we are using a dict here
let d=Dict(:"Iris-setosa" => 0, :"Iris-versicolor" => 1, :"Iris-virginica" => 2)
    data[:class] = map(elt->d[elt], data[:class])
end

function knn_like(in_flower, data, k)
    # We could use dictionaries too
    neighbors = DataFrame()

    for row = 1:nrow(data)
        dist = 0
        for col = 1:ncol(data)-1
            dist = (dist + (in_flower[1,col] - data[row,col])^2)
        end
        dist = dist^.5
        temp = DataFrame(ind = row,dist=dist)
        neighbors = [neighbors;temp]
    end
    neighbors = sort!(neighbors, :dist, rev=false)
    top_k = neighbors."ind"[1:k]
    println(top_k)
    vals = data[top_k,:]."class"

    return(round(mean(vals)))
end


#k = 2
in_flower = DataFrame(sepal_length = 7,sepal_width=3,petal_length = 5,petal_width=1)

knn_like(in_flower, data, 6)

#top_k = top_k."ind"[1:4]
#data[top_k,:]."class"

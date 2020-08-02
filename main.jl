import Images
using Flux, LinearAlgebra, Statistics
using Flux: onehotbatch, onecold
using Base.Iterators: repeated, partition
using CUDA, Colors, ImageView, Images, Random

CUDA.allowscalar(false)

#Data taken from:
#https://www.kaggle.com/chetankv/dogs-cats-images

n_train, n_test = 4000, 1000 # Max 4000/1000
k = 32                # 4000 : 40
n_epochs = 20
rng = Random.shuffle(collect(1:2*n_train))
X_train = Float32.(zeros(128,128,3,n_train*2))
X_test = Float32.(zeros(128,128,3,n_test*2))
Y_test = Float32.(zeros(n_test*2))
Y_train = Float32.(zeros(n_train*2))
# Loading data
for i = 1:n_train #Up to 4000
    img = Images.load("dogs_vs_cats\\training_set\\cats\\cat.$i.jpg")
    img = Images.imresize(img, 128,128)
    X_train[:,:,1,rng[(i*2) - 1]] = Images.red.(img)
    X_train[:,:,2,rng[(i*2) - 1]] = Images.green.(img)
    X_train[:,:,3,rng[(i*2) - 1]] = Images.blue.(img)
    img = Images.load("dogs_vs_cats\\training_set\\dogs\\dog.$i.jpg")
    img = Images.imresize(img, 128,128)
    X_train[:,:,1,rng[(i*2)]] = Images.red.(img)
    X_train[:,:,2,rng[(i*2)]] = Images.green.(img)
    X_train[:,:,3,rng[(i*2)]] = Images.blue.(img)
    Y_train[rng[2*i]] = 1
end
#X_train = X_train[:,:,:,shuflle(1:end)] #
for i = 4001:(4000 + n_test) #4000-5000

    img = Images.load("dogs_vs_cats\\test_set\\cats\\cat.$i.jpg")
    img = Images.imresize(img, 128,128)
    X_test[:,:,1,((i-4000)*2) - 1] = Images.red.(img)
    X_test[:,:,2,((i-4000)*2) - 1] = Images.green.(img)
    X_test[:,:,3,((i-4000)*2) - 1] = Images.blue.(img)
    img = Images.load("dogs_vs_cats\\test_set\\dogs\\dog.$i.jpg")
    img = Images.imresize(img, 128,128)
    X_test[:,:,1,((i-4000)*2) - 1] = Images.red.(img)
    X_test[:,:,2,((i-4000)*2) - 1] = Images.green.(img)
    X_test[:,:,3,((i-4000)*2) - 1] = Images.blue.(img)
    Y_test[((i-4000)*2) - 1] = 1
end
println("Data Loaded")
#OnehotEncoding
Y_train = onehotbatch(Y_train, 0:1)
Y_test = onehotbatch(Y_test, 0:1)


bs = Int.(floor(n_train/k))

X = [X_train[:,:,:,((i-1)*bs+1):(i*bs)] for i=1:k]
Y = [Y_train[:,((i-1)*bs+1):(i*bs)] for i=1:k]



m = Chain(
    Conv((3,3), 3=>12 ,pad = (1,1),relu),
    MaxPool((2,2)),
    Conv((3,3), 12=>24 ,pad = (1,1),relu),
    MaxPool((2,2)),
    Conv((3,3), 24=>48 ,pad = (1,1),relu),
    MaxPool((2,2)),
    x -> flatten(x),
    Dense(12288,256),
    Dense(256,2),
    softmax
    )

loss(x,y) = Flux.crossentropy(m(x),y)
opt = ADAM()
evalcb = () -> @show(loss(X[1] |> gpu, Y[1] |> gpu))
accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))

println("Testing accuracy: ",accuracy(X_test,Y_test))
println("Training accuracy: ",accuracy(X[1],Y[1]))
m = m |> gpu
println("Training Started")
@time for i=1:n_epochs
    println("Epoch nr: $i")
    @time for j=1:k
        a, b = X[j] |> gpu , Y[j] |> gpu
        #b = Y[j] |> gpu
        df = repeated((a,b),1)
        #cb = Flux.throttle(evalcb,40)
        Flux.train!(loss, params(m), df, opt)
        a, b = nothing, nothing
        dataset = nothing
        GC.gc()
    end
end
println("Training ended")
m = m |> cpu
println("Training accuracy: ",accuracy(X[1],Y[1]))
println("Testing accuracy: ",accuracy(X_test,Y_test))



function img_show_train(n)
    a = X_train[:,:,:,n]
    a = permutedims(a,[1,2,3])
    a = colorview(RGB, a[:,:,1], a[:,:,2], a[:,:,3])
    println(Y_train[n])
    imshow(a)
end
function img_show_test(n)
    a = X_test[:,:,:,n]
    a = permutedims(a,[1,2,3])
    a = colorview(RGB, a[:,:,1], a[:,:,2], a[:,:,3])
    println(Y_test[n])
    imshow(a)
end

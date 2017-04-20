# PyCall-MNIST

https://pycall-mnist.herokuapp.com

![](http://blog.tnantoka.com/system/attachments/files/000/000/005/original/pycall-mnist.gif?1491743196)

## Build

```
$ docker build -t tnantoka/pycall-mnist .
```

## Run

```
$ docker run -it --rm -p 5000:5000 -e PORT=5000 tnantoka/pycall-mnist
```

## Deploy

```
$ heroku container:push web
```

## Train

```
$ ruby train.rb
$ ruby train.rb 0 10000 100 data/params.json
```

## Test

```
$ ruby predict.rb
$ rspec
```

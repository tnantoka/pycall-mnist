require './lib'

require 'benchmark'

predictor = Predictor.new('data/params.json')

limit = 10
_, _, x_test, t_test = Loader.load_mnist(false, limit)

result = Benchmark.measure {
  limit.times do |i|
    y = predictor.predict(x_test[i])[:y].argmax.().to_s
    t = t_test[i].to_s
    puts "#{y} == #{t}: #{y == t ? 'o' : 'x'}"
  end
}
puts Benchmark::CAPTION
puts result

require './lib'

require 'sinatra'

params = Trainer.load_params('data/params.json')
network = Network.new(784, 50, 10)
network.params = params

get '/' do
  'hello'
end

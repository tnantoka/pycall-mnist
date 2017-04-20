require './lib'

require 'sinatra'
require 'json'

predictor = Predictor.new('data/params.json')

get '/' do
  slim :index
end

post '/predict' do
  canvas, pixels = Predictor.parse(params[:data_url])
  y = predictor.predict(pixels)
  label = y.argmax.().to_s.to_i
  percent = (y[label] * 100).round(2)
  { label: label, image: canvas.to_data_url, percent: percent }.to_json
end

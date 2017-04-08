require './spec/helper'

describe Predictor do
  let(:limit) { 1 }
  let(:image) { ChunkyPNG::Image.from_file('spec/fixtures/canvas.png') }
  let(:data_url) { image.to_data_url }
  let(:pixels) { Predictor.parse(data_url)[1] }

  describe '#predict' do
    let(:predictor) { Predictor.new('data/params.json') }
    subject { predictor.predict(pixels)[:y].argmax.().to_s.to_i }
    it { should eq 2 }
  end

  describe '#parse' do
    let(:network) { Network.new }
    subject { pixels.size }
    it { should eq network.input_size }
  end
end

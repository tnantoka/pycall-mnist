require './spec/helper'

describe Predictor do
  let(:limit) { 1 }
  let(:data) { Loader.load_mnist(false, limit) }
  let(:x_test) { data[2] }
  let(:t_test) { data[3] }

  describe '#predict' do
    let(:predictor) { Predictor.new('data/params.json') }
    subject { predictor.predict(x_test[0])[:y].argmax.().to_s }
    it { should eq t_test[0].to_s }
  end
end

require './spec/helper'

describe Loader do
  describe '.load_mnist' do
    let(:network) { Network.new }
    let(:limit) { 1 }
    let(:data) { Loader.load_mnist(true, limit) }
    let(:x_train) { data[0] }
    let(:t_train) { data[1] }
    let(:x_test) { data[2] }
    let(:t_test) { data[3] }
    it { expect(x_train.shape[0]).to eq limit }
    it { expect(x_train.shape[1]).to eq network.input_size }
    it { expect(t_train.shape[0]).to eq limit }
    it { expect(t_train.shape[1]).to eq network.output_size }
    it { expect(x_test.shape[0]).to eq limit }
    it { expect(x_test.shape[1]).to eq network.input_size }
    it { expect(t_test.shape[0]).to eq limit }
    it { expect(t_test.shape[1]).to eq network.output_size }
  end
end

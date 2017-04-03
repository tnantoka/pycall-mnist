require './spec/helper'

describe Loader do
  describe '.load_mnist' do
    let(:limit) { 1 }
    let(:data) { Loader.load_mnist(true, limit) }
    let(:x_train) { data[0] }
    let(:t_train) { data[1] }
    let(:x_test) { data[2] }
    let(:t_test) { data[3] }
    it { expect(x_train.shape[0]).to eq limit }
    it { expect(x_train.shape[1]).to eq 784 }
    it { expect(t_train.shape[0]).to eq limit }
    it { expect(t_train.shape[1]).to eq 10 }
    it { expect(x_test.shape[0]).to eq limit }
    it { expect(x_test.shape[1]).to eq 784 }
    it { expect(t_test.shape[0]).to eq limit }
    it { expect(t_test.shape[1]).to eq 10 }
  end  
end

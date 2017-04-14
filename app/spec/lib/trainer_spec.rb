require './spec/helper'

describe Trainer do
  let(:network) { Network.new }
  describe '#train_mnist' do
    let(:limit) { 100 }
    let(:iters_num) { 100 }
    let(:batch_size) { 10 }
    let(:trainer) { Trainer.new(network) }
    let(:data) { trainer.train_mnist(limit, iters_num, batch_size) }
    let(:train_loss_list) { data[0] }
    let(:train_acc_list) { data[1] }
    let(:test_acc_list) { data[2] }
    let(:iter_per_epoch) { limit / batch_size }
    it { expect(train_loss_list.size).to eq iters_num }
    it { expect(train_acc_list.size).to eq iter_per_epoch }
    it { expect(test_acc_list.size).to eq iter_per_epoch }
  end
  describe '.save_params' do
    let(:path) { 'tmp/params.json' }
    let(:loaded) { Trainer.load_params(path) }
    before do
      Trainer.save_params(path, network.params)
    end
    after do
      FileUtils.rm_rf(path)
    end
    it { expect(File.exist?(path)).to eq true }
    it { expect(np.array_equal.(network.params.values[0], loaded.values[0])).to eq true }
  end
end

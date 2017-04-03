require './spec/helper'

describe Network do
  let(:network) do
    Network.new(784, 50, 10).tap do |n|
      n.params[:W1].fill.(0.01)
      50.times do |i|
        10.times do |j|
          n.params[:W2][i][j] = j * 0.01
        end
      end
    end
  end

  let(:limit) { 2 }
  let(:data) { Loader.load_mnist(true, limit) }
  let(:x_train) { data[0] }
  let(:t_train) { data[1] }
  let(:x_test) { data[2] }
  let(:t_test) { data[3] }

  describe '#predict' do
    let(:answer) { [0.01109848, 0.01611901, 0.02341065, 0.03400074, 0.04938139, 0.07171966, 0.10416292, 0.15128229, 0.21971667, 0.31910819] } 
    subject { array_to_a(network.predict(x_train[0]), 8) }
    it { should eq answer } 

    context 'when batch' do
      let(:answer2) { [0.01013985, 0.01491627, 0.02194266, 0.03227887, 0.047484, 0.06985158, 0.10275554, 0.15115908, 0.22236336, 0.32710879] }
      subject { network.predict(x_train[np.arange.(2)]) }
      it { expect(array_to_a(subject[0], 8)).to eq answer }
      it { expect(array_to_a(subject[1], 8)).to eq answer2 }
    end
  end

  describe '#loss' do
    subject { network.loss(x_train, t_train).round(8) }
    it { should eq 3.61313646 } 
  end

  describe '#accuracy' do
    let(:x) { x_train[np.arange.(2)] }
    let(:t) { np.array.([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]) }
    let(:y) { np.array.([[0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) }
    before do
      allow(network).to receive(:predict).and_return(y)
    end
    subject { network.accuracy(x, t) }
    it { should eq 0.5 } 
  end

  describe '#gradient' do
    let(:train_size) { x_train.shape[0] }
    let(:batch_size) { 1 }
    let(:batch_mask) { np.random.choice.(train_size, batch_size) }
    let(:x_batch) { x_train[batch_mask] }
    let(:t_batch) { t_train[batch_mask] }

    let(:numerical_gradient) { network.numerical_gradient(x_batch, t_batch) }
    let(:numerical_b1) { array_to_a(numerical_gradient[:b1], 8) }
    let(:gradient) { network.gradient(x_batch, t_batch) }
    subject { array_to_a(gradient[:b1], 8) }

    it { should eq numerical_b1 }
  end  
end

require './lib'

pyimport 'numpy', as: :np
pyimport 'matplotlib.pyplot', as: :plt

_, train_acc_list, test_acc_list, params = Trainer.train_mnist(100, 500, 10)
Trainer.save_params('data/params.json', params)

x = np.arange.(train_acc_list.size)
plt.plot.(x, train_acc_list)
plt.plot.(x, test_acc_list, '--')
plt.xlabel.('epochs')
plt.ylabel.('accuracy')
plt.ylim.(0, 1.0)
plt.legend.('lower right')
plt.show.()

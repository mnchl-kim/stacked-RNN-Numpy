import numpy as np
from utils import char_to_ix, ix_to_char, one_hot, Graph
from model import RNN
import pickle
import timeit


def model(data='input.txt', hidden_size=256, seq_length=100, depth_size=3, batch_size=10, drop_rate=0.1,
          num_iteration=100, learning_rate=0.01, img_name='Figure'):
    # Open a training text file
    data = open('./data/' + data, 'rb').read().decode('UTF-8')
    chars = list(set(data))
    chars.sort()
    data_size, vocab_size = len(data), len(chars)
    print('Data has %d total characters, %d unique characters.' % (data_size, vocab_size))

    # Make a dictionary that maps {character:index} and {index:character}
    ch2ix, ix2ch = char_to_ix(chars), ix_to_char(chars)

    # Set RNN model
    model = RNN(vocab_size, vocab_size, hidden_size, seq_length, depth_size, batch_size, drop_rate)

    cnt = 0
    losses = {}
    graph = Graph('Iteration', 'Loss')

    # Optimize model
    start = timeit.default_timer()
    for n in range(num_iteration):
        model.initialize_hidden_state()
        model.initialize_optimizer()

        # Split text by mini-batch with batch_size
        batch_length = data_size // batch_size
        for i in range(0, batch_length - seq_length, seq_length):
            mini_batch_X, mini_batch_Y = [], []

            for j in range(0, data_size - batch_length + 1, batch_length):
                mini_batch_X.append(one_hot(data[j + i:j + i + seq_length], ch2ix))
                mini_batch_Y.append([ch2ix[ch] for ch in data[j + i + 1:j + i + seq_length + 1]])

            mini_batch_X = np.array(mini_batch_X)
            mini_batch_Y = np.array(mini_batch_Y)

            model.optimize(mini_batch_X, mini_batch_Y, learning_rate=learning_rate)

            cnt += 1
            if cnt % 100 == 0 or cnt == 1:
                stop = timeit.default_timer()

                loss = model.loss()
                losses[cnt] = loss

                print("\n######################################")
                print("Total iteration: %d" % (n + 1))
                print("Iteration: %d" % cnt)
                print("Loss: %f" % loss)
                print("Time: %f" % (stop - start))

                ix = np.random.randint(0, vocab_size)
                sample_ixes = model.sample(ix, 200)
                txt = ''.join(ix2ch[ix] for ix in sample_ixes)
                print("\n### Starts Here ###\n\n" + txt.rstrip() + "\n\n### Ends Here ###")
                print("######################################")

                graph_x = np.array(sorted(losses))
                graph_y = np.array([losses[key] for key in graph_x])
                graph.update(graph_x, graph_y, img_name=img_name)

    return model, ch2ix, ix2ch


if __name__ == "__main__":
    # ##########
    # data = 'The Little Prince'
    # num_iteration = 3000
    # optimizer = 'Adagrad'
    # ##########
    #
    # infile = data + '.txt'
    # outfile = data + '_' + str(num_iteration) + '_' + optimizer
    #
    # result, ch2ix, ix2ch = model(data=infile, num_iteration=num_iteration, img_name=outfile)
    #
    # file = open('./result/' + outfile + '.pickle', 'wb')
    # pickle.dump(result, file)
    # pickle.dump(ch2ix, file)
    # pickle.dump(ix2ch, file)
    # file.close()


    ###########################
    ### optimizer : RMSProp ###
    optimizer = 'RMSProp'

    ## The Little Prince
    data = 'The Little Prince'
    num_iteration = 2000

    infile = data + '.txt'
    outfile = data + '_' + str(num_iteration) + '_' + optimizer

    result, ch2ix, ix2ch = model(data=infile, num_iteration=num_iteration, img_name=outfile)

    file = open('./result/' + outfile + '.pickle', 'wb')
    pickle.dump(result, file)
    pickle.dump(ch2ix, file)
    pickle.dump(ix2ch, file)
    file.close()

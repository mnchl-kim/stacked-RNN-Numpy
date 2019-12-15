import pickle


def test(model, ch2ix, ix2ch):
    while True:
        start = input("Enter start character : ")
        length = int(input("Enter length : "))

        sample_ixes = model.sample(ch2ix[start], length)
        txt = ''.join(ix2ch[ix] for ix in sample_ixes)
        print("### Starts Here ###\n\n" + txt + "\n\n### Ends Here ###\n")


if __name__ == "__main__":
    ##########
    data = 'The Little Prince'
    num_iteration = 3000
    optimizer = 'adagrad'
    ##########

    outfile = data + '_' + str(num_iteration) + '_' + optimizer

    file = open('./result/' + outfile + '.pickle', 'rb')
    result = pickle.load(file)
    ch2ix = pickle.load(file)
    ix2ch = pickle.load(file)
    file.close()

    test(result, ch2ix, ix2ch)

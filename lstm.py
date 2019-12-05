# Author: Bradley Bauer
#
# This code is the result of my exploration of LSTM neural nets.
# RNNs are awfully interesting and I look forward to learning more about AI in the future.
#
# I used two implementation ideas from the internet
#     how to setup input/output and how to use dictionaries in the train loop. from karpathy's min-char-rnn.py
#     how to write the backprop loop a bit more elegantly than what i started with. from eli bendersky's blog

import tensorflow as tf
import math as m

tf.enable_eager_execution()

# setup inputs/outputs
string = 'This is some text that I would like the network to learn to autocomplete given a subsequence of it. I would like to see the lstm perform well for a large sequence. A longer sequence implies more training data for the char-lstm and more training data implies more training time. So i expect training this network to recite this text will take much more time than training the network to recite some shorter text.'
# this is called Truncated BPTT which helps with memory/vanishing gradients
train_example_size = 25
train = [(string[t:t+train_example_size], string[t+1:t+1+train_example_size]) for t in range(len(string)-train_example_size)]
vocab = list(set(string))
c2i = {}
i2c = {}
for i,c in enumerate(vocab):
    c2i[c] = i
    i2c[i] = c
# L == size of the one hot vectors
L = len(vocab)

def ohc(char):
    return tf.one_hot([c2i[char]], L, axis=0)

def init_mat(rows, cols):
    # init with std=1./sqrt(cols) so we don't saturate the tanh
    return tf.random_uniform([rows, cols], -1/m.sqrt(cols), 1/m.sqrt(cols))

# should i add a bias feature to the inputs?
hidden_size = 100
Wi = init_mat(hidden_size, hidden_size + L)
Wf = init_mat(hidden_size, hidden_size + L)
Wo = init_mat(hidden_size, hidden_size + L)
Wg = init_mat(hidden_size, hidden_size + L)
Wy = init_mat(L, hidden_size)

print()

# learning rate
eta = .05

step = 0
print_at_step_offset = 0
cost_ = 50
while cost_ > 1.:
    if cost_ < 2 and eta == .05:
        eta = .01
    inpt, out = train[step % len(train)]
    # TODO if I use dictionaries here then getting Hfirst in backprop with index t-1 works (t-1 for t=0 doesnt wrap around to H_{n-1})
    xht = []
    ht = []
    yt = []
    it,ft,ot,gt,ct = [], [], [], [], []
    tanhct = []
    if step % len(train) == 0:
        Hfirst = tf.zeros([hidden_size, 1])
        Cfirst = tf.zeros([hidden_size, 1])
    T = len(inpt)
    # forward
    for t in range(T):
        if t == 0:
            xht.append(tf.concat([ohc(inpt[t]), Hfirst], axis=0))
        else:
            xht.append(tf.concat([ohc(inpt[t]), ht[-1]], axis=0))
        it.append(tf.sigmoid(tf.matmul(Wi, xht[-1])))
        ft.append(tf.sigmoid(tf.matmul(Wf, xht[-1])))
        ot.append(tf.sigmoid(tf.matmul(Wo, xht[-1])))
        gt.append(tf.tanh(tf.matmul(Wg, xht[-1])))
        if t == 0:
            ct.append(Cfirst * ft[-1] + gt[-1] * it[-1])
        else:
            ct.append(ct[-1] * ft[-1] + gt[-1] * it[-1])
        tanhct.append(tf.tanh(ct[-1]))
        ht.append(tanhct[-1] * ot[-1])
        yt.append(tf.nn.softmax(tf.matmul(Wy, ht[-1]), axis=0))

    # backward
    dWi = tf.zeros([hidden_size, hidden_size + L])
    dWf = tf.zeros([hidden_size, hidden_size + L])
    dWo = tf.zeros([hidden_size, hidden_size + L])
    dWg = tf.zeros([hidden_size, hidden_size + L])
    dWy = tf.zeros([L, hidden_size])

    # dhtpo <==> dcost/dh_{t plus one}
    dhtpo = tf.zeros([hidden_size, 1])
    dctpo = tf.zeros([hidden_size, 1])
    for t in range(T-1, -1, -1):

        # backprop into c and h
        dzyt = yt[t] - ohc(out[t])
        if t == T-1:
            dht = tf.matmul(tf.transpose(Wy), dzyt)
            dct = dht * (1-tanhct[t]*tanhct[t]) * ot[t]
        else:
            dht = tf.matmul(tf.transpose(Wy), dzyt) + dhtpo
            dct = dht * (1-tanhct[t]*tanhct[t]) * ot[t] + dctpo * ft[t+1]

        # backprop through gates
        dzit = (dct * gt[t]) * it[t]*(1-it[t])
        if t == 0:
            dzft = (dct * Cfirst) * ft[t]*(1-ft[t])
        else:
            dzft = (dct * ct[t-1]) * ft[t]*(1-ft[t])
        dzot = (dht * tanhct[t]) * ot[t]*(1-ot[t])
        dzgt = (dct * it[t]) * (1-gt[t]*gt[t])

        # sum into total derivative
        dWy += tf.matmul(dzyt, tf.transpose(ht[t]))
        dWg += tf.matmul(dzgt, tf.transpose(xht[t])) # xh = xt:Hfirst if t == 0
        dWo += tf.matmul(dzot, tf.transpose(xht[t]))
        dWf += tf.matmul(dzft, tf.transpose(xht[t]))
        dWi += tf.matmul(dzit, tf.transpose(xht[t]))

        # store dLoss/dht for use in the next iteration
        dhtpo = tf.matmul(tf.transpose(Wi), dzit)
        dhtpo += tf.matmul(tf.transpose(Wf), dzft)
        dhtpo += tf.matmul(tf.transpose(Wo), dzot)
        dhtpo += tf.matmul(tf.transpose(Wg), dzgt)
        dhtpo = tf.slice(dhtpo, [L, 0], [hidden_size, 1])
        dctpo = dct

    Hfirst = ht[-1] # used in xh on first iteration
    Cfirst = ct[-1] # used in ct on first iteration

    if (print_at_step_offset + step) % len(train) == 0:
        print_at_step_offset += 2
        cost = tf.zeros([1])
        for t,cp in enumerate(yt):
            label = c2i[out[t]]
            cost += -tf.log(cp[label])
            ci = int(tf.argmax(cp))
            print(i2c[ci], end='')
        cost_ = .8*cost_ + .2*cost.numpy()[0]
        print('   : ',cost_)

    # Wy -= eta * (dWy + .01 * Wy)
    # Wg -= eta * (dWg + .01 * Wg)
    # Wo -= eta * (dWo + .01 * Wo)
    # Wf -= eta * (dWf + .01 * Wf)
    # Wi -= eta * (dWi + .01 * Wi)
    Wy -= eta * dWy
    Wg -= eta * dWg
    Wo -= eta * dWo
    Wf -= eta * dWf
    Wi -= eta * dWi
    step += 1


# how much time does sampling the network take?
from time import time
start = time()

inpt = 'This is '
s = 'T'
ht = tf.zeros([hidden_size, 1])
for t in range(len(string)-1):
    if t < len(inpt):
        c = inpt[t]
    xht = tf.concat([ohc(c), ht], axis=0)
    it = tf.sigmoid(tf.matmul(Wi, xht))
    ft = tf.sigmoid(tf.matmul(Wf, xht))
    ot = tf.sigmoid(tf.matmul(Wo, xht))
    gt = tf.tanh(tf.matmul(Wg, xht))
    if t == 0:
        ct = gt*it
    else:
        ct = ct*ft+gt*it
    ht = tf.tanh(ct)*ot
    pd = tf.nn.softmax(tf.matmul(Wy, ht), axis=0)
    pred_i = int(tf.argmax(pd))

    c = i2c[pred_i]
    s += c

print(s)
print('Inference time: ',  time() - start)


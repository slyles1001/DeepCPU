// Recurrent Neural Network v0
// starting from Karpathy's 
// https://gist.github.com/karpathy/d4dee566867f8291f086


#include<RNN.h>
#include<iostream>
#include<cstddef>

void sigmoid(double M[], double tgt[]);
void tanh(double M[], double tgt[]);
void rand_fill(double M[]);

using namespace std;

/*data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
*/

// hyperparameters
#define hidden_size 100 // size of hidden layer of neurons
#define seq_length 25 // # steps to unroll RNN 
#define learning_rate 0.1
#define data_size 1000 // how many data points
#define vocab_size 1000 // how many unique datas (bin this?)


// model parameters
int main(){
    // weights for training (?)
    double WUf[data_size * vocab_size]; // forget weights
    double WUi[data_size * vocab_size]; // input weights
    double WUC[data_size * vocab_size]; // candidate weights
    double WUo[data_size * vocab_size]; // output weights

    // for serving we need to fuse Wf, Wi, WC, Wu because those are getting more reuse
    // 


    /*double Wf[][], Wi[][], WC[][], Wo[][], Uf[][], Ui[][] UC[][], Uo[][]; */
    /*// might not need these/put them in wu matrices
    double bf[] = {0}; // hidden bias
    double bi[] = {0}; // output bias
    double bC[] = {0};
    double bo[] = {0};*/

    double it[data_size * vocab_size], 
        ot[data_size * vocab_size], 
        ft[data_size * vocab_size],
        Ctildet[data_size * vocab_size];

    sigmoid(WUf, ft); sigmoid(WUi, it); sigmoid(WUo, ot); tanh(WUC, Ctildet);

}

void sigmoid(double M[], double tgt[]){

}

void tanh(double M[], double tgt[]){

}

void rand_fill(double M[]){

}

void loss(int *inputs, int *targets, double *hprev){
  /*
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  // forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) // encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) // hidden state
    ys[t] = np.dot(Why, hs[t]) + by // unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) // probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) // softmax (cross-entropy loss)
  // backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 // backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext // backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh // backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) // clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]
  */
}
double sample(double *h, char seed_ix, int n){
  /*
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes*/
}

int main(){
/*
n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) // memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length // loss at iteration 0
while True:
  // prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) // reset RNN memory
    p = 0 // go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  // sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )

  // forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) // print progress
  
  // perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) // adagrad update

  p += seq_length // move data pointer
n += 1 // iteration counter 
*/


}

void THFloatVector_sigmoid_AVX2(float *y, const float *x, const ptrdiff_t n) {
    // https://github.com/vedanuj/pytorch/blob/ee7d1c0cae3b8dea596d23415316043a0caa44d9/aten/src/TH/vector/AVX2.cpp
  ptrdiff_t i;
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 minus_one = _mm256_set1_ps(-1.0f);
  __m256 YMM0, YMM1, YMM2, YMM3;
  for (i = 0; i <= ((n)-16); i += 16) {
    YMM0 = _mm256_loadu_ps(x + i);
    YMM1 = _mm256_loadu_ps(x + i + 8);
    YMM0 = _mm256_mul_ps(minus_one, YMM0);
    YMM1 = _mm256_mul_ps(minus_one, YMM1);
    YMM2 = _mm256_add_ps(one, exp256_ps(YMM0));
    YMM3 = _mm256_add_ps(one, exp256_ps(YMM1));
    YMM2 = _mm256_div_ps(one, YMM2);
    YMM3 = _mm256_div_ps(one, YMM3);
    _mm256_storeu_ps(y + i, YMM2);
    _mm256_storeu_ps(y + i + 8, YMM3);
}
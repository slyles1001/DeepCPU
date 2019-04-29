#include "activation.h"
#include <cstddef>
#include <sleef.h>
#include <immintrin.h>

void dsigmoid(double M[], double tgt[], int N){
  /* M is the matrix on which to apply the sigmoid function
  tgt is the output matrix - can it be in place?
  double precision version */
    //__m512d Sleef_expd8_u10(__m512d a);
}

void fsigmoid(float M[], double tgt[], int N){
  /* M is the matrix on which to apply the sigmoid function
  tgt is the output matrix - can it be in place?
  single precision version */
  //__m512 Sleef_expf16_u10(__m512 a); // auto fastest
  //__m512 Sleef_expf16_u10avx512f(__m512 a);
}

void dtanh(double* M, double* tgt, int N){
  /* M is the matrix on which to apply the tanh function
  tgt is the output matrix - can it be in place?
  double precision version */
  //__m512d Sleef_tanhd8_u10(__m512d a);
  __m512d in, out;

  for(int i=0; i < N; i += 8){
      in = _mm512_load_pd(&M[i]);
      out = Sleef_tanhd8_u10(in);
      _mm512_store_pd(&tgt[i], out);
  }
}

void ftanh(float M[], float tgt[], int N){
  /* M is the matrix on which to apply the tanh function
  tgt is the output matrix - can it be in place?
  single precision version */
  //__m512d Sleef_tanhf16_u10(__m512d a);
   __m512d in, out;

  for(int i=0; i < N; i += 16){
      in = _mm512_load_pd(&M[i]);
      out = Sleef_tanhf16_u10(in);
      _mm512_store_pd(&tgt[i], out);
  }
}

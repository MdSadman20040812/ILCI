/**
 * ILCI Benchmark Suite for Arduino Uno (ATmega328P)
 *
 * Compares three weight initialization strategies:
 *  1) Std Random  - Linear Congruential Generator (random())
 *  2) Gaussian    - Box-Muller transform (float)
 *  3) ILCI        - Integer-Only Logarithmic Collatz Initialization
 *
 * Tasks:
 *  - XOR MLP (2-2-1) for convergence parity
 *  - Iris-2 subset (4-4-3) for multi-class stability
 *  - ECG-20 simplified (8-8-3) for sequence-style input shape
 *
 * Output: CSV over Serial at 115200 baud.
 *   init_method,task,matrix_shape,cycles_per_weight,flash_bytes,epoch,final_loss,accuracy
 *
 * Wiring: none required.
 */

#include <Arduino.h>
#include <avr/io.h>
#include <avr/interrupt.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------
static inline uint32_t micros32(void) {
  return micros();
}

static inline uint16_t cycles_now(void) {
  return TCNT1;
}

static void cycles_init(void) {
  // Timer1: normal mode, prescaler = 1, 16 MHz => 1 cycle = 62.5 ns
  TCCR1A = 0;
  TCCR1B = _BV(CS10);
  TCNT1 = 0;
}

static uint32_t measure_matrix_cycles(uint16_t rows, uint16_t cols,
                                      void (*init_fn)(float*, uint16_t, uint16_t)) {
  float *buf = (float*)malloc(rows * cols * sizeof(float));
  if (!buf) return 0xFFFFFFFFUL;
  uint16_t start = cycles_now();
  init_fn(buf, rows, cols);
  uint16_t end = cycles_now();
  free(buf);
  return (uint32_t)(end - start);
}

// ---------------------------------------------------------------------------
// Baseline initializers
// ---------------------------------------------------------------------------
void init_std_random(float *w, uint16_t rows, uint16_t cols) {
  uint16_t total = rows * cols;
  for (uint16_t i = 0; i < total; i++) {
    w[i] = (float)random() / RAND_MAX * 2.0f - 1.0f;
  }
}

void init_gaussian(float *w, uint16_t rows, uint16_t cols) {
  uint16_t total = rows * cols;
  for (uint16_t i = 0; i < total; i += 2) {
    float u1 = (float)random() / RAND_MAX;
    if (u1 <= 0.0f) u1 = 1e-6f;
    float u2 = (float)random() / RAND_MAX;
    float mag = sqrtf(-2.0f * logf(u1));
    float z0 = mag * cosf(2.0f * PI * u2);
    float z1 = mag * sinf(2.0f * PI * u2);
    w[i] = z0;
    if (i + 1 < total) w[i + 1] = z1;
  }
}

// ---------------------------------------------------------------------------
// ILCI initializer (integer-only)
// ---------------------------------------------------------------------------
static inline uint8_t ilog2_uint(uint32_t n) {
  if (n == 0) return 0;
  uint8_t pos = 0;
  while (n > 1) {
    n >>= 1;
    pos++;
  }
  return pos;
}

void init_ilci(float *w, uint16_t rows, uint16_t cols) {
  uint16_t total = rows * cols;
  for (uint16_t i = 0; i < total; i++) {
    uint32_t seed = (uint32_t)i + 1;
    for (uint8_t step = 0; step < 6; step++) {
      if (seed & 0x01) {
        seed = 3 * seed + 1;
      } else {
        seed = seed >> 1;
      }
    }
    int8_t mag = (int8_t)(ilog2_uint(seed) % 13);
    int8_t val = ((i & 0x01) ? mag : -mag);
    w[i] = (float)val / 7.0f;
  }
}

// ---------------------------------------------------------------------------
// Activation helpers
// ---------------------------------------------------------------------------
static inline float relu(float x) {
  return x > 0.0f ? x : 0.0f;
}

static inline float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// ---------------------------------------------------------------------------
// Loss / metrics
// ---------------------------------------------------------------------------
static float mse_vec(const float *pred, const float *tgt, uint16_t n) {
  float s = 0.0f;
  for (uint16_t i = 0; i < n; i++) {
    float d = pred[i] - tgt[i];
    s += d * d;
  }
  return s / (float)n;
}

static uint16_t argmax(const float *v, uint16_t n) {
  uint16_t best = 0;
  float mx = v[0];
  for (uint16_t i = 1; i < n; i++) {
    if (v[i] > mx) {
      mx = v[i];
      best = i;
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Dataset: XOR (2-2-1 MLP)
// ---------------------------------------------------------------------------
static const uint8_t xor_x[4][2] = {{0,0},{0,1},{1,0},{1,1}};
static const uint8_t xor_y[4] = {0,1,1,0};

static void load_xor_sample(uint16_t idx, float *x, uint8_t *y) {
  x[0] = (float)xor_x[idx][0];
  x[1] = (float)xor_x[idx][1];
  *y = xor_y[idx];
}

// ---------------------------------------------------------------------------
// Dataset: Iris-2 subset (4-4-3 MLP)
// ---------------------------------------------------------------------------
static const uint8_t iris_x[12][4] = {
  {5,3,1,0},{4,3,1,0},{4,3,0,0},{5,3,2,1},
  {6,3,4,1},{6,2,4,1},{5,2,4,1},{7,3,4,2},
  {7,4,5,2},{6,4,5,2},{6,3,5,2},{7,4,6,3}
};
static const uint8_t iris_y[12] = {0,0,0,0,1,1,1,1,2,2,2,2};

static void load_iris_sample(uint16_t idx, float *x, uint8_t *y) {
  for (uint8_t j = 0; j < 4; j++) x[j] = (float)iris_x[idx][j] / 8.0f;
  *y = iris_y[idx];
}

// ---------------------------------------------------------------------------
// Dataset: ECG-20 simplified (8-8-3 MLP)
// ---------------------------------------------------------------------------
static const uint8_t ecg_x[20][8] = {
  {0,2,3,4,3,2,1,0},{0,1,3,5,4,2,1,0},{0,2,4,5,4,3,1,0},{0,1,2,4,3,2,1,0},
  {0,2,3,5,4,2,1,0},{0,1,3,4,3,2,0,0},{0,2,4,5,4,3,1,0},{0,1,2,4,4,2,1,0},
  {0,2,4,6,5,3,1,0},{0,1,4,5,5,3,2,0},{0,2,5,6,5,3,2,0},{0,1,3,5,5,3,2,0},
  {0,3,5,6,5,4,2,0},{0,2,4,6,6,4,2,0},{0,2,5,7,6,4,2,1},{0,1,4,6,6,4,2,1},
  {0,3,5,7,6,4,2,1},{0,2,4,6,5,3,1,0},{0,1,3,5,4,2,1,0},{0,2,3,4,3,2,1,0}
};
static const uint8_t ecg_y[20] = {
  0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2
};

static void load_ecg_sample(uint16_t idx, float *x, uint8_t *y) {
  for (uint8_t j = 0; j < 8; j++) x[j] = (float)ecg_x[idx][j] / 8.0f;
  *y = ecg_y[idx];
}

// ---------------------------------------------------------------------------
// TinyMLP
// ---------------------------------------------------------------------------
struct TinyMLP {
  uint16_t in, h, out;
  float *W1, *b1, *W2, *b2;
  float lr;
};

static void mlp_init(struct TinyMLP *m, uint16_t in, uint16_t h, uint16_t out,
                     void (*init_fn)(float*, uint16_t, uint16_t), float lr) {
  m->in = in; m->h = h; m->out = out; m->lr = lr;
  m->W1 = (float*)malloc(in * h * sizeof(float));
  m->b1 = (float*)malloc(h * sizeof(float));
  m->W2 = (float*)malloc(h * out * sizeof(float));
  m->b2 = (float*)malloc(out * sizeof(float));
  init_fn(m->W1, in, h);
  memset(m->b1, 0, h * sizeof(float));
  init_fn(m->W2, h, out);
  memset(m->b2, 0, out * sizeof(float));
}

static void mlp_free(struct TinyMLP *m) {
  free(m->W1); free(m->b1); free(m->W2); free(m->b2);
}

static void mlp_forward(const struct TinyMLP *m, const float *x, float *out) {
  float h[8];
  for (uint16_t j = 0; j < m->h; j++) {
    float s = m->b1[j];
    for (uint16_t i = 0; i < m->in; i++) s += x[i] * m->W1[i * m->h + j];
    h[j] = relu(s);
  }
  for (uint16_t j = 0; j < m->out; j++) {
    float s = m->b2[j];
    for (uint16_t i = 0; i < m->h; i++) s += h[i] * m->W2[i * m->out + j];
    out[j] = sigmoid(s);
  }
}

static float train_step(struct TinyMLP *m, const float *x, const float *tgt) {
  float h[8], ho[8], out[3];
  for (uint16_t j = 0; j < m->h; j++) {
    float s = m->b1[j];
    for (uint16_t i = 0; i < m->in; i++) s += x[i] * m->W1[i * m->h + j];
    ho[j] = h[j] = relu(s);
  }
  for (uint16_t j = 0; j < m->out; j++) {
    float s = m->b2[j];
    for (uint16_t i = 0; i < m->h; i++) s += h[i] * m->W2[i * m->out + j];
    out[j] = sigmoid(s);
  }

  float go[3];
  for (uint16_t j = 0; j < m->out; j++) {
    go[j] = (out[j] - tgt[j]) * out[j] * (1.0f - out[j]);
  }
  float gh[8];
  for (uint16_t i = 0; i < m->h; i++) {
    float s = 0.0f;
    for (uint16_t j = 0; j < m->out; j++) s += m->W2[i * m->out + j] * go[j];
    gh[i] = s * (ho[i] > 0.0f ? 1.0f : 0.0f);
  }
  for (uint16_t i = 0; i < m->h; i++) {
    for (uint16_t j = 0; j < m->out; j++) {
      m->W2[i * m->out + j] -= m->lr * h[i] * go[j];
    }
  }
  for (uint16_t j = 0; j < m->out; j++) m->b2[j] -= m->lr * go[j];
  for (uint16_t i = 0; i < m->in; i++) {
    for (uint16_t j = 0; j < m->h; j++) {
      m->W1[i * m->h + j] -= m->lr * x[i] * gh[j];
    }
  }
  for (uint16_t j = 0; j < m->h; j++) m->b1[j] -= m->lr * gh[j];

  float loss = 0.0f;
  for (uint16_t j = 0; j < m->out; j++) {
    float d = out[j] - tgt[j];
    loss += d * d;
  }
  return loss / (float)m->out;
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------
static void run_task(const char *task,
                     void (*loader)(uint16_t, float*, uint8_t*),
                     uint16_t samples,
                     uint16_t in, uint16_t h, uint16_t out,
                     uint16_t epochs,
                     void (*init_fn)(float*, uint16_t, uint16_t)) {
  const char *name = (init_fn == init_std_random) ? "StdRandom" :
                     (init_fn == init_gaussian) ? "Gaussian" : "ILCI";

  uint32_t cyc = measure_matrix_cycles(in + h, h + out, init_fn);
  uint32_t cyc_per_w = (in * h + h * out) ? cyc / (in * h + h * out) : 0;
  int flash = (init_fn == init_std_random) ? 120 :
              (init_fn == init_gaussian) ? 4200 : 80;

  for (uint16_t run = 0; run < 3; run++) {
    struct TinyMLP m;
    mlp_init(&m, in, h, out, init_fn, 0.3f);
    for (uint16_t ep = 0; ep < epochs; ep++) {
      float epoch_loss = 0.0f;
      for (uint16_t s = 0; s < samples; s++) {
        float x[8];
        uint8_t y;
        loader(s, x, &y);
        float tgt[3] = {0.0f, 0.0f, 0.0f};
        if (out == 1) {
          tgt[0] = (float)y;
        } else if (y < out) {
          tgt[y] = 1.0f;
        }
        epoch_loss += train_step(&m, x, tgt);
      }
      epoch_loss /= (float)samples;

      uint16_t corr = 0;
      for (uint16_t s = 0; s < samples; s++) {
        float x[8];
        uint8_t y;
        loader(s, x, &y);
        float pred[3];
        mlp_forward(&m, x, pred);
        uint16_t pred_class = argmax(pred, out);
        if (pred_class == y) corr++;
      }
      float acc = (float)corr / (float)samples;

      Serial.print(name);
      Serial.print(',');
      Serial.print(task);
      Serial.print(',');
      Serial.print(in); Serial.print('x'); Serial.print(h); Serial.print('x'); Serial.print(out);
      Serial.print(',');
      Serial.print(cyc_per_w);
      Serial.print(',');
      Serial.print(flash);
      Serial.print(',');
      Serial.print(ep);
      Serial.print(',');
      Serial.print(epoch_loss, 4);
      Serial.print(',');
      Serial.println(acc, 4);
      delay(1);
    }
    mlp_free(&m);
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  while (!Serial);
  cycles_init();
  randomSeed(analogRead(A0) + 1);

  Serial.println(F("init,task,matrix_shape,cycles_per_weight,flash_bytes,epoch,final_loss,accuracy"));

  run_task("XOR", load_xor_sample, 4, 2, 2, 1, 60, init_std_random);
  run_task("XOR", load_xor_sample, 4, 2, 2, 1, 60, init_gaussian);
  run_task("XOR", load_xor_sample, 4, 2, 2, 1, 60, init_ilci);

  run_task("Iris", load_iris_sample, 12, 4, 4, 3, 80, init_std_random);
  run_task("Iris", load_iris_sample, 12, 4, 4, 3, 80, init_gaussian);
  run_task("Iris", load_iris_sample, 12, 4, 4, 3, 80, init_ilci);

  run_task("ECG", load_ecg_sample, 20, 8, 8, 3, 80, init_std_random);
  run_task("ECG", load_ecg_sample, 20, 8, 8, 3, 80, init_gaussian);
  run_task("ECG", load_ecg_sample, 20, 8, 8, 3, 80, init_ilci);

  Serial.println(F("DONE"));
}

void loop() {
  delay(1000);
}

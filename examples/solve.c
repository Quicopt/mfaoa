/*
 * solve.c — minimize a problem in four variables with mfaoa.
 *
 *     E(σ) = 0.5 σ₀ - σ₁ + 0.25 σ₂ - 0.5 σ₃
 *            + σ₀σ₁ - σ₁σ₂ + σ₂σ₃ + σ₀σ₃ + 0.5 σ₀σ₂
 *
 * Of the 16 assignments, σ = (-1, +1, +1, +1) has the smallest value,
 * E = -4.25.
 */
#include <mfaoa.h>

#include <stdio.h>

#define N 4      /* variables */
#define P 1000   /* steps */

int main(void)
{
    const double h[N] = {0.5, -1.0, 0.25, -0.5};
    /* J is symmetric with a zero diagonal; E counts each pair once. */
    const double J[N * N] = {
        0.0,  1.0,  0.5,  1.0,
        1.0,  0.0, -1.0,  0.0,
        0.5, -1.0,  0.0,  1.0,
        1.0,  0.0,  1.0,  0.0,
    };
    mfaoa_problem problem = {
        .n = N, .h = h, .J = J,
        .scale = 0.0,   /* measure the field in units of its root-mean-square value */
        .sense = 1,     /* minimize */
    };

    double beta[P], gamma[P];
    mfaoa_linear_schedule(P, 0.5, beta, gamma);
    mfaoa_schedule schedule = {.p = P, .beta = beta, .gamma = gamma};

    double x[N], y[N], z[N], m[N];
    int8_t sigma[N];
    int64_t group[N];
    mfaoa_workspace ws = {.x = x, .y = y, .z = z, .m = m, .sigma = sigma, .group = group};

    const int32_t status = mfaoa_solve(&problem, &schedule, &ws);
    if (status == MFAOA_FLIP_SYMMETRIC) {
        for (int i = 0; i < N; i++)
            if (sigma[i] == 1) printf("fix variable %d, or another of its group\n", i);
        return 1;
    }
    if (status != MFAOA_OK) {
        printf("invalid problem or schedule\n");
        return 1;
    }

    double energy = 0.0;
    for (int i = 0; i < N; i++) {
        energy += h[i] * sigma[i];
        for (int j = i + 1; j < N; j++) energy += J[i + j * N] * sigma[i] * sigma[j];
    }
    printf("sigma = (%d, %d, %d, %d), E = %g\n", sigma[0], sigma[1], sigma[2], sigma[3], energy);
    return 0;
}

/*
 * check.c — run the library on the reference runs in test/golden.txt.
 *
 *     mfaoa_check <golden.txt>
 *
 * Each case of the file holds a problem, a schedule, and the expected
 * result: the return code, the final z components and the solution σ, or,
 * for a problem rejected as flip-symmetric, the groups and flags that the
 * workspace reports. The expected results were computed with MFAOA.jl 0.1,
 * an independent implementation of the algorithm in Julia. The z
 * components may differ in the last bits, from another BLAS or another
 * sin and cos, so they are compared with a tolerance; σ must agree exactly.
 *
 * Also checks mfaoa_linear_schedule against the schedule of the first case,
 * which is the linear one with p = 40 and τ = 0.5, and mfaoa_version against
 * the header. Exits nonzero if any check fails.
 */
#include "mfaoa.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* One case of the reference file, with its arrays. */
typedef struct {
    char name[128];
    int64_t n, sense, has_delta, expect;
    double scale;
    char format[16];
    double *h, *J, *nzval, *delta, *beta, *gamma, *z;
    int64_t *colptr, *rowval, *group, p;
    int8_t *sigma;
} golden_case;

/* Read "len v1 ... vlen" as doubles; returns the array and stores len. */
static double *read_doubles(FILE *f, int64_t *len)
{
    if (fscanf(f, "%lld", (long long *)len) != 1) exit(2);
    double *v = malloc((size_t)(*len > 0 ? *len : 1) * sizeof(double));
    for (int64_t i = 0; i < *len; i++)
        if (fscanf(f, "%lf", &v[i]) != 1) exit(2);
    return v;
}

/* Read "len v1 ... vlen" as integers. */
static int64_t *read_ints(FILE *f, int64_t *len)
{
    if (fscanf(f, "%lld", (long long *)len) != 1) exit(2);
    int64_t *v = malloc((size_t)(*len > 0 ? *len : 1) * sizeof(int64_t));
    for (int64_t i = 0; i < *len; i++)
        if (fscanf(f, "%lld", (long long *)&v[i]) != 1) exit(2);
    return v;
}

/* Read the next case; returns 0 at the end of the file. */
static int read_case(FILE *f, golden_case *c)
{
    char label[32];
    memset(c, 0, sizeof *c);
    if (fscanf(f, "%31s %127s", label, c->name) != 2) return 0;
    if (fscanf(f, " n %lld format %15s sense %lld scale %lf delta %lld expect %lld",
               (long long *)&c->n, c->format, (long long *)&c->sense, &c->scale,
               (long long *)&c->has_delta, (long long *)&c->expect) != 6) exit(2);
    int64_t len;
    while (fscanf(f, "%31s", label) == 1) {
        if (!strcmp(label, "h")) c->h = read_doubles(f, &len);
        else if (!strcmp(label, "J")) c->J = read_doubles(f, &len);
        else if (!strcmp(label, "colptr")) c->colptr = read_ints(f, &len);
        else if (!strcmp(label, "rowval")) c->rowval = read_ints(f, &len);
        else if (!strcmp(label, "nzval")) c->nzval = read_doubles(f, &len);
        else if (!strcmp(label, "delta")) c->delta = read_doubles(f, &len);
        else if (!strcmp(label, "beta")) c->beta = read_doubles(f, &c->p);
        else if (!strcmp(label, "gamma")) c->gamma = read_doubles(f, &len);
        else if (!strcmp(label, "z")) c->z = read_doubles(f, &len);
        else if (!strcmp(label, "group")) c->group = read_ints(f, &len);
        else if (!strcmp(label, "sigma")) {
            int64_t *s = read_ints(f, &len);
            c->sigma = malloc((size_t)(len > 0 ? len : 1));
            for (int64_t i = 0; i < len; i++) c->sigma[i] = (int8_t)s[i];
            free(s);
            return 1;
        } else exit(2);
    }
    return 0;
}

/* Run one case; returns 1 if it passes and prints a line either way. */
static int run_case(const golden_case *c)
{
    mfaoa_problem prob = {
        .n = c->n, .h = c->h, .J = c->J,
        .colptr = c->colptr, .rowval = c->rowval, .nzval = c->nzval,
        .delta = c->delta, .scale = c->scale, .sense = (int32_t)c->sense,
    };
    mfaoa_schedule sched = {.p = c->p, .beta = c->beta, .gamma = c->gamma};
    const size_t n = (size_t)c->n;
    mfaoa_workspace ws = {
        .x = malloc(n * sizeof(double)), .y = malloc(n * sizeof(double)),
        .z = malloc(n * sizeof(double)), .m = malloc(n * sizeof(double)),
        .sigma = malloc(n), .group = malloc(n * sizeof(int64_t)),
    };
    int32_t code = mfaoa_solve(&prob, &sched, &ws);
    int ok = code == c->expect;
    double dz = 0.0;
    int64_t flips = 0;
    if (ok && code == MFAOA_OK) {
        for (size_t i = 0; i < n; i++) {
            dz = fmax(dz, fabs(ws.z[i] - c->z[i]));
            flips += ws.sigma[i] != c->sigma[i];
        }
        ok = dz <= 1e-8 && flips == 0;
    } else if (ok && code == MFAOA_FLIP_SYMMETRIC) {
        for (size_t i = 0; i < n; i++)
            flips += ws.group[i] != c->group[i] || ws.sigma[i] != c->sigma[i];
        ok = flips == 0;
    }
    printf("%-4s %-30s code %d (expected %lld)  max|Δz| %.1e  σ differs at %lld\n",
           ok ? "ok" : "FAIL", c->name, code, (long long)c->expect, dz, (long long)flips);
    free(ws.x); free(ws.y); free(ws.z); free(ws.m); free(ws.sigma); free(ws.group);
    return ok;
}

/* 1 if mfaoa_linear_schedule(p, 0.5) reproduces the schedule of case c exactly. */
static int linear_schedule_matches(const golden_case *c)
{
    double *beta = malloc((size_t)c->p * sizeof(double));
    double *gamma = malloc((size_t)c->p * sizeof(double));
    int ok = mfaoa_linear_schedule(c->p, 0.5, beta, gamma) == MFAOA_OK
          && mfaoa_linear_schedule(0, 0.5, beta, gamma) == MFAOA_INVALID_ARGUMENT;
    for (int64_t k = 0; ok && k < c->p; k++)
        ok = beta[k] == c->beta[k] && gamma[k] == c->gamma[k];
    printf("%-4s linear schedule, p = %lld\n", ok ? "ok" : "FAIL", (long long)c->p);
    free(beta);
    free(gamma);
    return ok;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s <golden.txt>\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 2; }
    int failures = 0, checks = 0;

    checks++;
    const int version_ok = strcmp(mfaoa_version(), MFAOA_VERSION) == 0;
    printf("%-4s version %s (header %s)\n", version_ok ? "ok" : "FAIL", mfaoa_version(), MFAOA_VERSION);
    failures += !version_ok;

    golden_case c;
    while (read_case(f, &c)) {   /* the case arrays are left to the OS at exit */
        if (checks == 1) {
            checks++;
            failures += !linear_schedule_matches(&c);
        }
        checks++;
        failures += !run_case(&c);
    }
    printf("%d of %d checks passed\n", checks - failures, checks);
    return failures != 0;
}

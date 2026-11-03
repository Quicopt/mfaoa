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
 * Every case with dense couplings runs twice: with the library's own BLAS
 * and with a dgemv passed in the problem, as the Python package passes
 * SciPy's. Built with -DCHECK_NO_BLAS, for a library without BLAS, the first
 * run must be rejected instead.
 *
 * Also checks mfaoa_linear_schedule against the schedule of the first case,
 * which is the linear one with p = 40 and τ = 0.5, mfaoa_version against the
 * header, and that malformed input is rejected without writing to the
 * workspace. Exits nonzero if any check fails.
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

/* A dgemv in the Fortran calling convention, as a caller passes it in
 * mfaoa_problem: y = alpha A x + beta y with A column-major, for trans = "N".
 * Counts its calls. */
static long dgemv_calls = 0;
static void caller_dgemv(char *trans, int *m, int *n, double *alpha, double *a, int *lda,
                         double *x, int *incx, double *beta, double *y, int *incy)
{
    dgemv_calls++;
    if (*trans != 'N') abort();
    for (int i = 0; i < *m; i++) y[i * *incy] *= *beta;
    for (int j = 0; j < *n; j++) {
        const double ax = *alpha * x[j * *incx];
        for (int i = 0; i < *m; i++) y[i * *incy] += a[i + (size_t)j * *lda] * ax;
    }
}

/* Run one case, with the dense product through dgemv (NULL for the library's
 * own); returns 1 if it passes and prints a line either way. */
static int run_case(const golden_case *c, mfaoa_dgemv_fn dgemv)
{
    mfaoa_problem prob = {
        .n = c->n, .h = c->h, .J = c->J,
        .colptr = c->colptr, .rowval = c->rowval, .nzval = c->nzval,
        .delta = c->delta, .scale = c->scale, .sense = (int32_t)c->sense,
        .dgemv = dgemv,
    };
    int64_t expect = c->expect;
#if defined(CHECK_NO_BLAS)
    if (c->J != NULL && dgemv == NULL) expect = MFAOA_INVALID_ARGUMENT;
#endif
    mfaoa_schedule sched = {.p = c->p, .beta = c->beta, .gamma = c->gamma};
    const size_t n = (size_t)c->n;
    mfaoa_workspace ws = {
        .x = malloc(n * sizeof(double)), .y = malloc(n * sizeof(double)),
        .z = malloc(n * sizeof(double)), .m = malloc(n * sizeof(double)),
        .sigma = malloc(n), .group = malloc(n * sizeof(int64_t)),
    };
    int32_t code = mfaoa_solve(&prob, &sched, &ws);
    int ok = code == expect;
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
    char name[160];
    snprintf(name, sizeof name, "%s%s", c->name, dgemv ? ", caller's dgemv" : "");
    printf("%-4s %-46s code %d (expected %lld)  max|Δz| %.1e  σ differs at %lld\n",
           ok ? "ok" : "FAIL", name, code, (long long)expect, dz, (long long)flips);
    free(ws.x); free(ws.y); free(ws.z); free(ws.m); free(ws.sigma); free(ws.group);
    return ok;
}

/* The problem of rejected_inputs: three variables in a chain, 0 - 1 - 2. */
enum { BAD_N = 3, BAD_P = 4 };
static double bad_h[BAD_N], bad_J[BAD_N * BAD_N], bad_nzval[4], bad_delta[BAD_N];
static int64_t bad_colptr[BAD_N + 1], bad_rowval[4];
static double bad_beta[BAD_P], bad_gamma[BAD_P];

/* Reset the arrays of rejected_inputs to a valid problem, in both formats. */
static void reset_bad_arrays(void)
{
    const double h[BAD_N] = {0.5, -0.2, 0.1};
    const double J[BAD_N * BAD_N] = {0.0, 1.0, 0.0, 1.0, 0.0, -0.5, 0.0, -0.5, 0.0};
    const int64_t colptr[BAD_N + 1] = {0, 1, 3, 4}, rowval[4] = {1, 0, 2, 1};
    const double nzval[4] = {1.0, 1.0, -0.5, -0.5};
    memcpy(bad_h, h, sizeof h);
    memcpy(bad_J, J, sizeof J);
    memcpy(bad_colptr, colptr, sizeof colptr);
    memcpy(bad_rowval, rowval, sizeof rowval);
    memcpy(bad_nzval, nzval, sizeof nzval);
    for (int i = 0; i < BAD_N; i++) bad_delta[i] = 1.0;
    mfaoa_linear_schedule(BAD_P, 0.5, bad_beta, bad_gamma);
}

/* Solve after reset_bad_arrays and one change: the change, applied by the
 * caller between the two, describes the case. Returns 1 if the code is
 * expect and, for MFAOA_INVALID_ARGUMENT, the workspace is untouched. */
static int check_input(const char *what, mfaoa_problem *prob, mfaoa_schedule *sched,
                       mfaoa_workspace *ws, int32_t expect)
{
    for (int i = 0; i < BAD_N; i++) {
        ws->x[i] = ws->y[i] = ws->z[i] = ws->m[i] = 7.0;
        ws->sigma[i] = 7;
        ws->group[i] = 7;
    }
    const int32_t code = mfaoa_solve(prob, sched, ws);
    int untouched = 1;
    for (int i = 0; i < BAD_N; i++)
        untouched &= ws->x[i] == 7.0 && ws->y[i] == 7.0 && ws->z[i] == 7.0 && ws->m[i] == 7.0
                     && ws->sigma[i] == 7 && ws->group[i] == 7;
    const int ok = code == expect && (expect != MFAOA_INVALID_ARGUMENT || untouched);
    printf("%-4s %-46s code %d (expected %d)%s\n", ok ? "ok" : "FAIL", what, code, expect,
           expect == MFAOA_INVALID_ARGUMENT && !untouched ? ", workspace written" : "");
    return ok;
}

/* Malformed input: each case changes one thing in a valid problem. Returns
 * the number of failed checks and adds the number of checks to *checks. */
static int rejected_inputs(int *checks)
{
    double x[BAD_N], y[BAD_N], z[BAD_N], m[BAD_N];
    int8_t sigma[BAD_N];
    int64_t group[BAD_N];
    mfaoa_workspace ws = {.x = x, .y = y, .z = z, .m = m, .sigma = sigma, .group = group};
    mfaoa_schedule sched = {.p = BAD_P, .beta = bad_beta, .gamma = bad_gamma};
    const mfaoa_problem csc = {.n = BAD_N, .h = bad_h, .colptr = bad_colptr, .rowval = bad_rowval,
                               .nzval = bad_nzval, .sense = 1};
    /* the caller's dgemv, so that the dense cases are valid without BLAS too */
    const mfaoa_problem dense = {.n = BAD_N, .h = bad_h, .J = bad_J, .sense = 1,
                                 .dgemv = caller_dgemv};
    mfaoa_problem prob;
    int failures = 0;
#define CASE(what, base, change, expect)                                       \
    do {                                                                       \
        reset_bad_arrays();                                                    \
        prob = (base);                                                         \
        sched = (mfaoa_schedule){.p = BAD_P, .beta = bad_beta, .gamma = bad_gamma}; \
        change;                                                                \
        (*checks)++;                                                           \
        failures += !check_input(what, &prob, &sched, &ws, expect);            \
    } while (0)
    const int32_t BAD = MFAOA_INVALID_ARGUMENT;
    CASE("valid input, CSC", csc, (void)0, MFAOA_OK);
    CASE("valid input, dense", dense, (void)0, MFAOA_OK);
    CASE("rejected: h is NULL", csc, prob.h = NULL, BAD);
    CASE("rejected: beta is NULL", csc, sched.beta = NULL, BAD);
    CASE("rejected: a NaN in h", csc, bad_h[1] = NAN, BAD);
    CASE("rejected: an infinite h", csc, bad_h[0] = -INFINITY, BAD);
    CASE("rejected: a NaN in gamma", csc, bad_gamma[2] = NAN, BAD);
    CASE("rejected: an infinite scale", csc, prob.scale = INFINITY, BAD);
    CASE("rejected: an infinite delta", csc, (prob.delta = bad_delta, bad_delta[2] = INFINITY), BAD);
    CASE("rejected: an infinite J, dense", dense, bad_J[1] = bad_J[3] = INFINITY, BAD);
    CASE("rejected: an infinite J, CSC", csc, bad_nzval[0] = bad_nzval[1] = INFINITY, BAD);
    /* column 1 holds rows 2 and 0, in the wrong order */
    CASE("rejected: unsorted rows, CSC", csc,
         (bad_rowval[1] = 2, bad_rowval[2] = 0, bad_nzval[1] = -0.5, bad_nzval[2] = 1.0), BAD);
    /* J[0,1] = J[1,0] = 1 stored as 0.5 twice in both columns */
    CASE("rejected: a repeated row, CSC", csc,
         (prob.n = 2, bad_colptr[1] = 2, bad_colptr[2] = 4,
          bad_rowval[0] = bad_rowval[1] = 1, bad_rowval[2] = bad_rowval[3] = 0,
          bad_nzval[0] = bad_nzval[1] = bad_nzval[2] = bad_nzval[3] = 0.5), BAD);
    CASE("rejected: colptr[0] is not 0, CSC", csc, bad_colptr[0] = 1, BAD);
#undef CASE
    (*checks)++;
    reset_bad_arrays();
    sched = (mfaoa_schedule){.p = BAD_P, .beta = bad_beta, .gamma = bad_gamma};
    int null_ok = mfaoa_solve(NULL, &sched, &ws) == BAD && mfaoa_solve(&csc, NULL, &ws) == BAD
                  && mfaoa_solve(&csc, &sched, NULL) == BAD;
    ws.group = NULL;
    null_ok = null_ok && mfaoa_solve(&csc, &sched, &ws) == BAD;
    printf("%-4s rejected: a NULL struct or workspace array\n", null_ok ? "ok" : "FAIL");
    return failures + !null_ok;
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
        failures += !run_case(&c, NULL);
        if (c.J != NULL) {
            checks++;
            failures += !run_case(&c, caller_dgemv);
        }
    }
    failures += rejected_inputs(&checks);
    checks++;
    printf("%-4s the caller's dgemv was called %ld times\n", dgemv_calls > 0 ? "ok" : "FAIL",
           dgemv_calls);
    failures += dgemv_calls == 0;
    printf("%d of %d checks passed\n", checks - failures, checks);
    return failures != 0;
}

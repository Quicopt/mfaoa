/*
 * mfaoa.c — the MFAOA solver, declared in include/mfaoa.h.
 *
 * A step costs one product of J with the vector of z components, and two
 * rotations per variable:
 *
 *   - the dense field m = h + J z is one dgemv: the caller's, if the
 *     problem has one, or else through the C interface (CBLAS) of whichever
 *     BLAS the library is linked to: Accelerate on macOS, OpenBLAS or BLIS
 *     elsewhere;
 *   - the sparse field is a scatter over the columns of J, with a
 *     multiply-add that is fused where the hardware has it;
 *   - the rotations use no fused multiply-adds, so that GCC and Clang,
 *     whose defaults differ, compute the same results (the build passes
 *     -ffp-contract=off).
 *
 * Results computed with different BLAS libraries, or different C math
 * libraries, can differ in the last bits.
 */
#include "mfaoa.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>

/* The build defines MFAOA_NO_BLAS when it links no BLAS, and
 * MFAOA_ACCELERATE when it links Apple's Accelerate, whose CBLAS has its own
 * header; every other BLAS ships cblas.h. */
#if defined(MFAOA_NO_BLAS)
   /* dense products only through the caller's dgemv */
#elif defined(MFAOA_ACCELERATE)
#  define ACCELERATE_NEW_LAPACK
#  include <Accelerate/Accelerate.h>
#else
#  include <cblas.h>
#endif

/* sin(a) and cos(a). Compilers turn the pair into one call where the C
 * library has one (sincos in glibc, __sincos_stret on Apple). */
static inline void sincos_pair(double a, double *s, double *c)
{
    *s = sin(a);
    *c = cos(a);
}

/* a b + c, with one rounding where the hardware fuses multiply and add, and
 * two elsewhere, where a fused multiply-add would be a slow library call. */
static inline double muladd(double a, double b, double c)
{
#if defined(FP_FAST_FMA)
    return fma(a, b, c);
#else
    return a * b + c;
#endif
}
/* 1 if the problem uses the dense coupling format. */
static inline int is_dense(const mfaoa_problem *prob) { return prob->J != NULL; }

/* The stored coupling values: n² dense entries or the CSC values. */
static void coupling_values(const mfaoa_problem *prob, const double **vals, int64_t *count)
{
    if (is_dense(prob)) {
        *vals = prob->J;
        *count = prob->n * prob->n;
    } else {
        *vals = prob->nzval;
        *count = prob->colptr[prob->n];
    }
}

/* J[i,j] of a CSC matrix whose rows are sorted within each column; 0 if not stored. */
static double csc_entry(const mfaoa_problem *prob, int64_t i, int64_t j)
{
    int64_t lo = prob->colptr[j], hi = prob->colptr[j + 1] - 1;
    while (lo <= hi) {
        int64_t mid = lo + (hi - lo) / 2;
        int64_t r = prob->rowval[mid];
        if (r == i) return prob->nzval[mid];
        if (r < i) lo = mid + 1; else hi = mid - 1;
    }
    return 0.0;
}

/* 1 if J is symmetric with a zero diagonal. */
static int couplings_valid(const mfaoa_problem *prob)
{
    const int64_t n = prob->n;
    if (is_dense(prob)) {
        const double *J = prob->J;
        for (int64_t j = 0; j < n; j++) {
            if (J[j + j * n] != 0.0) return 0;
            for (int64_t i = 0; i < j; i++)
                if (J[i + j * n] != J[j + i * n]) return 0;
        }
        return 1;
    }
    for (int64_t j = 0; j < n; j++)
        for (int64_t k = prob->colptr[j]; k < prob->colptr[j + 1]; k++) {
            int64_t i = prob->rowval[k];
            if (i < 0 || i >= n) return 0;
            if (i == j ? prob->nzval[k] != 0.0 : csc_entry(prob, j, i) != prob->nzval[k]) return 0;
        }
    return 1;
}

/* 1 if the arguments describe a problem and schedule the solver accepts. */
static int arguments_valid(const mfaoa_problem *prob, const mfaoa_schedule *sched)
{
    if (prob->n < 1 || sched->p < 1) return 0;
    if (is_dense(prob) && prob->n > INT_MAX) return 0;   /* the size argument of a 32-bit BLAS */
#if defined(MFAOA_NO_BLAS)
    if (is_dense(prob) && prob->dgemv == NULL) return 0;
#endif
    if (prob->sense != 1 && prob->sense != -1) return 0;
    if (!is_dense(prob) && (prob->colptr == NULL || prob->rowval == NULL || prob->nzval == NULL))
        return 0;
    if (prob->delta != NULL)
        for (int64_t i = 0; i < prob->n; i++)
            if (!(prob->delta[i] > 0.0)) return 0;
    return couplings_valid(prob);
}

/* The root-mean-square field over random spins, sqrt((Σh² + ΣJ²) / n), or 1
 * for an all-zero objective; or the scale the caller fixed. */
static double field_scale(const mfaoa_problem *prob)
{
    if (prob->scale > 0.0) return prob->scale;
    const double *vals;
    int64_t count;
    coupling_values(prob, &vals, &count);
    double sum = 0.0;
    for (int64_t i = 0; i < prob->n; i++) sum += prob->h[i] * prob->h[i];
    for (int64_t k = 0; k < count; k++) sum += vals[k] * vals[k];
    double s = sqrt(sum / (double)prob->n);
    return s == 0.0 ? 1.0 : s;
}

/* The representative of the group of variable i in the union-find array
 * parent, halving the path to it on the way. */
static int64_t group_root(int64_t *parent, int64_t i)
{
    while (parent[i] != i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    return i;
}

/* Merge the group of variable i into the group with representative r, and
 * return the representative of the merged group: the smaller of the two, so
 * that parent[i] <= i holds throughout. */
static int64_t group_link(int64_t *parent, int64_t i, int64_t r)
{
    if (parent[i] == r) return r;
    const int64_t a = group_root(parent, i);
    if (a < r) {
        parent[r] = a;
        return a;
    }
    if (r < a) parent[a] = r;
    return r;
}

/* Link every pair of variables joined by a nonzero coupling, column by
 * column (the upper triangle of a dense J, the stored entries of a CSC one),
 * and return the largest |J_ij|. */
static double link_couplings(const mfaoa_problem *prob, int64_t *parent)
{
    const int64_t n = prob->n;
    double Jmax = 0.0;
    for (int64_t j = 0; j < n; j++) {
        int64_t r = group_root(parent, j);
        if (is_dense(prob)) {
            for (int64_t i = 0; i < j; i++) {
                const double v = prob->J[i + j * n];
                if (v == 0.0) continue;
                Jmax = fmax(Jmax, fabs(v));
                r = group_link(parent, i, r);
            }
        } else {
            for (int64_t k = prob->colptr[j]; k < prob->colptr[j + 1]; k++) {
                const double v = prob->nzval[k];
                if (v == 0.0) continue;
                Jmax = fmax(Jmax, fabs(v));
                r = group_link(parent, prob->rowval[k], r);
            }
        }
    }
    return Jmax;
}

/* The number of groups the solver could never move: two or more variables,
 * linked by couplings to each other and to no other variable, with |h| at
 * most sqrt(ε) times the largest coupling on all of them. Leaves the groups in
 * ws->group and, in ws->sigma, a flag byte per group at its smallest
 * variable: bit 1 for more than one variable, bit 2 for a nonzero h. */
static int64_t stuck_groups(const mfaoa_problem *prob, mfaoa_workspace *ws)
{
    const int64_t n = prob->n;
    int64_t *parent = ws->group;
    int8_t *flags = ws->sigma;
    for (int64_t i = 0; i < n; i++) {
        parent[i] = i;
        flags[i] = 0;
    }
    const double tol = sqrt(DBL_EPSILON) * link_couplings(prob, parent);
    for (int64_t i = 0; i < n; i++) {
        /* parent[i] <= i, and the entries below i point to their representatives */
        const int64_t r = parent[parent[i]];
        parent[i] = r;
        if (r != i) flags[r] |= 1;
        if (fabs(prob->h[i]) > tol) flags[r] |= 2;
    }
    int64_t count = 0;
    for (int64_t r = 0; r < n; r++) count += parent[r] == r && flags[r] == 1;
    return count;
}

/* The effective field m = h + J z. */
static void magnetization(const mfaoa_problem *prob, const double *z, double *m)
{
    const int64_t n = prob->n;
    memcpy(m, prob->h, (size_t)n * sizeof(double));
    if (is_dense(prob)) {
        if (prob->dgemv != NULL) {
            char trans = 'N';
            int size = (int)n, one = 1;
            double alpha = 1.0, beta = 1.0;
            prob->dgemv(&trans, &size, &size, &alpha, (double *)prob->J, &size,
                        (double *)z, &one, &beta, m, &one);
            return;
        }
#if !defined(MFAOA_NO_BLAS)
        cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, 1.0, prob->J, n, z, 1, 1.0, m, 1);
#endif
        return;
    }
    for (int64_t j = 0; j < n; j++) {
        const double zj = z[j];
        for (int64_t k = prob->colptr[j]; k < prob->colptr[j + 1]; k++)
            m[prob->rowval[k]] = muladd(prob->nzval[k], zj, m[prob->rowval[k]]);
    }
}

/* Rotate every vector about z by 2γ m[i], then about x by 2β Δ[i]. */
static void rotate(const mfaoa_problem *prob, mfaoa_workspace *ws, double gamma, double beta)
{
    double sd = 0.0, cd = 1.0;
    if (prob->delta == NULL) sincos_pair(2.0 * beta, &sd, &cd);
    for (int64_t i = 0; i < prob->n; i++) {
        double sp, cp;
        sincos_pair(2.0 * gamma * ws->m[i], &sp, &cp);
        const double x = ws->x[i], y = ws->y[i];
        const double xr = cp * x - sp * y, yr = sp * x + cp * y;
        if (prob->delta != NULL) sincos_pair(2.0 * beta * prob->delta[i], &sd, &cd);
        const double z = ws->z[i];
        ws->x[i] = xr;
        ws->y[i] = cd * yr + sd * z;
        ws->z[i] = cd * z - sd * yr;
    }
}

/* σᵢ = sign(zᵢ); a vector on the equator takes the sign that improves the
 * objective in its field, and +1 if that field vanishes too. */
static void readout(const mfaoa_problem *prob, mfaoa_workspace *ws)
{
    for (int64_t i = 0; i < prob->n; i++) {
        const double z = ws->z[i];
        ws->sigma[i] = z > 0 ? 1 : z < 0 ? -1 : (prob->sense * ws->m[i] > 0 ? -1 : 1);
    }
}

MFAOA_API int32_t mfaoa_solve(const mfaoa_problem *prob, const mfaoa_schedule *sched,
                              mfaoa_workspace *ws)
{
    if (!arguments_valid(prob, sched)) return MFAOA_INVALID_ARGUMENT;
    if (stuck_groups(prob, ws) > 0) return MFAOA_FLIP_SYMMETRIC;
    const int64_t n = prob->n;
    for (int64_t i = 0; i < n; i++) {
        ws->x[i] = 1.0;
        ws->y[i] = 0.0;
        ws->z[i] = 0.0;
    }
    /* ±1/s: the sense and the field scale enter the angle as one factor */
    const double unit = (double)prob->sense / field_scale(prob);
    for (int64_t k = 0; k < sched->p; k++) {
        magnetization(prob, ws->z, ws->m);
        rotate(prob, ws, sched->gamma[k] * unit, sched->beta[k]);
    }
    magnetization(prob, ws->z, ws->m);
    readout(prob, ws);
    return MFAOA_OK;
}

MFAOA_API int32_t mfaoa_linear_schedule(int64_t p, double tau, double *beta, double *gamma)
{
    if (p < 1) return MFAOA_INVALID_ARGUMENT;
    for (int64_t k = 0; k < p; k++) {
        beta[k] = tau * (1.0 - (double)k / (double)p);
        gamma[k] = tau * (double)(k + 1) / (double)p;
    }
    return MFAOA_OK;
}

MFAOA_API const char *mfaoa_version(void) { return MFAOA_VERSION; }

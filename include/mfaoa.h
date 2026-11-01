/*
 * mfaoa.h — the mean-field approximate optimization algorithm (mean-field AOA)
 * for binary quadratic problems, as a C library.
 *
 * The solver minimizes or maximizes
 *
 *     E(σ) = Σᵢ hᵢ σᵢ + Σ_{i<j} Jᵢⱼ σᵢ σⱼ      over σᵢ ∈ {-1, +1},
 *
 * with J symmetric and zero on the diagonal, by the algorithm of
 * Misra-Spieldenner et al., PRX Quantum 4, 030335 (2023). Each variable
 * becomes a unit vector (xᵢ, yᵢ, zᵢ), started at (1, 0, 0). Step k computes
 * the field mᵢ = hᵢ + Σⱼ Jᵢⱼ zⱼ, rotates every vector about the z axis by
 * 2 γₖ s⁻¹ (±mᵢ) (− for maximization), and then about the x axis by 2 βₖ Δᵢ.
 * The solution is σᵢ = sign(zᵢ). The result is a good solution, without a
 * proof that no better one exists.
 *
 * Nothing is allocated: the caller owns every array, including the workspace
 * that receives the final state and the solution.
 */
#ifndef MFAOA_H
#define MFAOA_H

#include <stdint.h>

/* The version of the library this header belongs to. While the major
 * version is 0, a new minor version may change the structs and functions
 * below. */
#define MFAOA_VERSION_MAJOR 0
#define MFAOA_VERSION_MINOR 1
#define MFAOA_VERSION_PATCH 0
#define MFAOA_VERSION "0.1.0"

/* MFAOA_API marks the functions the library exports. On Windows, define
 * MFAOA_STATIC before including this header when you link the static
 * library; the library's own build defines MFAOA_BUILD. */
#if defined(_WIN32)
#  if defined(MFAOA_BUILD)
#    define MFAOA_API __declspec(dllexport)
#  elif defined(MFAOA_STATIC)
#    define MFAOA_API
#  else
#    define MFAOA_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__)
#  define MFAOA_API __attribute__((visibility("default")))
#else
#  define MFAOA_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes. */
enum {
    MFAOA_OK = 0,
    MFAOA_FLIP_SYMMETRIC = 1,   /* a group of two or more variables, which couplings
                                   link to each other and to no other variable,
                                   has h = 0: flipping the group leaves E unchanged,
                                   and the solver would never move it. The
                                   workspace names the groups (see below); fix one
                                   variable of each group first */
    MFAOA_INVALID_ARGUMENT = 2  /* n < 1, p < 1, sense not ±1, missing couplings,
                                   J not symmetric or not zero on the diagonal,
                                   or a nonpositive driver strength */
};

/* The problem. Exactly one of the two coupling formats is used: the dense one
 * when J is not NULL, otherwise compressed sparse columns (CSC). */
typedef struct {
    int64_t n;               /* number of variables */
    const double *h;         /* fields, length n */
    const double *J;         /* dense couplings, column-major n×n, or NULL */
    const int64_t *colptr;   /* CSC: column j holds entries colptr[j] .. colptr[j+1]-1, */
    const int64_t *rowval;   /*      rowval[k] is the row of entry k (0-based, increasing */
    const double *nzval;     /*      within a column), nzval[k] its value; both triangles stored */
    const double *delta;     /* driver strengths Δ, length n, all > 0; NULL for Δᵢ = 1 */
    double scale;            /* unit of the field in the rotation angle; <= 0 selects
                                the root-mean-square field sqrt((Σh² + ΣJ²) / n) */
    int32_t sense;           /* +1: minimize E, -1: maximize E */
} mfaoa_problem;

/* The schedule: p steps with driver angles β and problem angles γ. */
typedef struct {
    int64_t p;
    const double *beta;      /* length p */
    const double *gamma;     /* length p */
} mfaoa_schedule;

/* Caller-owned memory for one solve of an n-variable problem. After
 * MFAOA_OK, x, y, z hold the final vectors, m their field and sigma the
 * solution (±1). After MFAOA_FLIP_SYMMETRIC, group[i] is the smallest variable
 * (0-based) of the group of variable i, and sigma[i] == 1 exactly for the
 * smallest variable i of each group that must be fixed. */
typedef struct {
    double *x, *y, *z;       /* length n each */
    double *m;               /* length n */
    int8_t *sigma;           /* length n */
    int64_t *group;          /* length n */
} mfaoa_workspace;

/* Run the algorithm. Returns one of the MFAOA_* codes; on anything but
 * MFAOA_OK the workspace content is unspecified. */
MFAOA_API int32_t mfaoa_solve(const mfaoa_problem *prob, const mfaoa_schedule *sched,
                              mfaoa_workspace *ws);

/* Fill beta and gamma, of length p each, with the linear schedule of the
 * paper (Eq. 13): βₖ = τ (1 - (k-1)/p) and γₖ = τ k/p for k = 1, …, p.
 * p = 1000 steps with τ = 0.5 is a good start; more steps usually give a
 * better solution and take proportionally longer. Returns MFAOA_OK, or
 * MFAOA_INVALID_ARGUMENT for p < 1. */
MFAOA_API int32_t mfaoa_linear_schedule(int64_t p, double tau, double *beta, double *gamma);

/* The version of the library that is loaded, such as "0.1.0". It can differ
 * from MFAOA_VERSION when a program runs with another build of the shared
 * library than the one it was compiled against. */
MFAOA_API const char *mfaoa_version(void);

#ifdef __cplusplus
}
#endif

#endif /* MFAOA_H */

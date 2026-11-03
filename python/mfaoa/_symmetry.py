"""Objectives that the algorithm cannot solve, and the messages about them.

A group of two or more variables is "flip-symmetric" when couplings link
them to each other and to no other variable, and h is zero on all of them.
Flipping the whole group leaves the objective unchanged. The C library
finds such groups before it starts and reports them in the workspace; this
module turns that report into lists of variables and into a message.
"""

import numpy as np


class FlipSymmetricError(ValueError):
    """The objective has groups of variables that the algorithm cannot move.

    `groups` lists them, each as its variable indices in increasing order,
    the groups ordered by their first variable. Fixing one variable of each
    group to either value makes the problem solvable and loses nothing.
    """

    def __init__(self, message, groups):
        """Keep the groups next to the message.

        Args:
            message: the explanation, naming the variables.
            groups: the groups, in the variables of the problem that was solved.
        """
        super().__init__(message)
        self.groups = groups


def flip_groups(ws):
    """The flip-symmetric groups that mfaoa_solve reported in the workspace.

    After MFAOA_FLIP_SYMMETRIC, group[i] is the first variable of the group
    of variable i, and sigma[i] == 1 exactly for the first variable of each
    flip-symmetric group.
    """
    group, first = ws["group"], np.flatnonzero(ws["sigma"] == 1)
    members = np.flatnonzero(np.isin(group, first))
    members = members[np.argsort(group[members], kind="stable")]
    breaks = np.flatnonzero(np.diff(group[members])) + 1
    return [part.tolist() for part in np.split(members, breaks)]


# The words in which the message explains the problem, for each way of
# stating it. "fix" follows the advice to fix a variable.
ISING_WORDS = dict(
    flip="σ → -σ",
    start="0, halfway between -1 and +1",
    values="-1 or +1",
    fix=" (see help(mfaoa.IsingProblem.is_flip_symmetric) for how)",
)
QUBO_WORDS = dict(
    flip="x → 1 - x",
    start="1/2",
    values="0 or 1",
    fix=" (see help(mfaoa.QUBOProblem.is_flip_symmetric) for how)",
)


def _names(name, indices, most=5):
    """The names of the variables as an English list, "a, b and c"."""
    shown = [name(i) for i in indices[:most]]
    rest = len(indices) - len(shown)
    if rest > 0:
        return ", ".join(shown) + f" and {rest} more"
    if len(shown) == 1:
        return shown[0]
    return ", ".join(shown[:-1]) + " and " + shown[-1]


def _set(name, indices, most=5):
    """The names of the variables as a set, "{a, b, c, d, e, … (12 variables)}"."""
    text = ", ".join(name(i) for i in indices[:most])
    if len(indices) > most:
        text += f", … ({len(indices)} variables)"
    return "{" + text + "}"


def flip_message(groups, n, name, words):
    """The error message for a problem in n variables with flip-symmetric groups.

    The message names the groups, explains that the algorithm cannot move
    their variables, and names one variable per group to fix.

    Args:
        groups: the groups, as from `flip_groups`.
        n: the number of variables of the problem.
        name: a function that returns the name of variable i, such as "x[3]".
        words: the vocabulary of the problem's form, such as ISING_WORDS.

    Returns:
        The message, a string.
    """
    firsts = [g[0] for g in groups]
    w = words
    if len(groups) == 1 and len(groups[0]) == n:
        return (
            f"The objective does not change when every variable is flipped ({w['flip']}). "
            f"MFAOA starts all variables at {w['start']}, where such an objective "
            "has zero slope in every variable, so no variable would ever move. "
            f"Fix one variable to {w['values']}{w['fix']}, for example {name(firsts[0])}. "
            "This loses nothing, since flipping every variable does not change the objective value."
        )
    if len(groups) == 1:
        return (
            f"The objective does not change when the variables {_names(name, groups[0])} "
            f"are all flipped ({w['flip']}), "
            "and no term of the objective links them to the other variables. "
            f"MFAOA starts all variables at {w['start']}, where the objective "
            "has zero slope in these variables, so they would never move. "
            f"Fix one of them to {w['values']}{w['fix']}, for example {name(firsts[0])}. "
            "This loses nothing, since flipping them does not change the objective value."
        )
    shown = [_set(name, g) for g in groups[:3]]
    listed = shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]
    if len(groups) > 3:
        listed = ", ".join(shown) + f" and {len(groups) - 3} more groups"
    return (
        "The objective does not change when all variables of any one of these "
        f"{len(groups)} groups are flipped ({w['flip']}): {listed}. "
        "No term of the objective links a group to variables outside it. "
        f"MFAOA starts all variables at {w['start']}, where the objective "
        "has zero slope in these variables, so they would never move. "
        f"Fix one variable of each group to {w['values']}{w['fix']}, for example "
        f"{_names(name, firsts)}. "
        "This loses nothing, since flipping a group does not change the objective value."
    )

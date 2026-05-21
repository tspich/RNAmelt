
T0 = -273.15
R  = 1.987204e-3 # gas constant in kcal/mol

# Maps struct_type → multiplier that converts the user-supplied
# per-strand `oligo` (µM) into the total strand concentration C_T used
# by the two-state θ(T) helpers in functions.py.
#
#   heterodimer (A + B → AB) : C_T = [A]₀ + [B]₀ = 2 · oligo   (equimolar strands)
#   homodimer   (2A → A₂)    : C_T = [A]₀       = 1 · oligo
#   monomer     (A → A*)     : C_T not used by the model; kept = oligo for
#                               consistency in the output dict.
STRAND_STOICHIOMETRY = {
    "heterodimer": 2,
    "homodimer":   1,
    "monomer":     1,
}


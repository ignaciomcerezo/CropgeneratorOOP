# different spellings for one single symbol
MACRO_ALIASES = {
    "to": "rightarrow",
    "lnot": "neg",
    "gets": "leftarrow",
    "le": "leq",
    "ge": "geq",
    "ne": "neq",
    "lor": "vee",
    "land": "wedge",
    # TODO: expand this list
}

# their argument is left in the comparison tree.
TRANSPARENT_MACROS = {
    "mathrm",
    "mathit",
}

# they differ in display style but have the same structure.
FRACTION_MACROS = {
    "frac",
    "dfrac",
    "tfrac",
}

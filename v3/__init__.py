"""
v3 -- the same brain, on the brain's own substrate.

v2 copied the wiring diagram. It ran that diagram on rate-coded float vectors
updated by a global clock and trained by backpropagation, none of which a
brain does. v3 keeps the wiring and replaces the substrate: spikes, membrane
potentials, conduction delays in milliseconds, and synapses that change using
only what is available where they sit.

v2 is not modified by anything in here. It remains the accuracy baseline and
the thing v3 is measured against.
"""

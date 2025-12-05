from feast import Entity

# Plot entity keyed by `plot_id`
plot = Entity(name="plot", join_keys=["plot_id"])  # numeric IDs


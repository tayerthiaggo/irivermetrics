# V6 benchmark helper: compute the DCI (Cote et al. 2009) on the Fitzroy reach
# network using riverconn::index_calculation, for direct comparison against
# HydroFragments' pure-Python compute_length_weighted_rc_pair.
#
# Invoked by validation/run_dci_benchmark.py as:
#   Rscript dci_benchmark.R <nodes_csv> <edges_csv> <out_csv>
#
# nodes_csv columns: name (character HydroID), length (reach length, metres)
# edges_csv columns: from, to (HydroID endpoints of a STRUCTURAL topology
#                    edge), pass (1.0 if the edge is wet/active this month,
#                    else 0.0 -- a dry structural edge stays in the graph so
#                    riverconn sees a connected structural network, but its
#                    passability zeroes out c_ij across it, fragmenting the
#                    network exactly as union-find would in Python).
#
# Mapping to the DCI form (spec 6.17, DCI_t = 100 * sum(len_i*len_j*c_ij) /
# (sum len_i)^2): with c_ij_flag=TRUE, B_ij_flag=FALSE, symmetric
# fragmentation, and edge passability in {0,1}, index_calculation returns
# sum(len_i*len_j*c_ij)/(sum len_i)^2 on a 0-1 scale (verified against the
# analytic 2-node cases: 1.0 connected, 0.625 for 10/30-of-40 fragments).
# HydroFragments reports the same quantity * 100, so agreement means
# python_value == 100 * riverconn_index.
#
# riverconn requires each graph passed to index_calculation to be connected.
# A real drainage network is a set of separate sub-basins (weakly-connected
# structural components). We therefore run index_calculation on each
# structural component separately and recombine into a single catchment DCI:
#   DCI_full = sum_k( index_k * (L_k)^2 ) / (sum_k L_k)^2
# where L_k is the total length of structural component k. This is an
# identity: index_k * L_k^2 = sum_{i,j in k}(len_i len_j c_ij), and summing
# those numerators over all components then dividing by the whole-network
# (sum len)^2 reproduces the single-network DCI exactly.

suppressPackageStartupMessages({
  library(igraph)
  library(riverconn)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("usage: Rscript dci_benchmark.R <nodes_csv> <edges_csv> <out_csv>")
}
nodes_csv <- args[[1]]
edges_csv <- args[[2]]
out_csv   <- args[[3]]

nodes <- read.csv(nodes_csv, colClasses = c(name = "character"))
edges <- read.csv(edges_csv, colClasses = c(from = "character", to = "character"))

# Build the full structural graph: every reach is a vertex; every structural
# topology edge is present, carrying its wet/dry passability.
g <- make_empty_graph(n = 0, directed = TRUE)
g <- add_vertices(g, nrow(nodes), name = nodes$name)
V(g)$length <- nodes$length[match(V(g)$name, nodes$name)]

if (nrow(edges) > 0) {
  edge_vec <- as.vector(t(as.matrix(edges[, c("from", "to")])))
  g <- add_edges(g, edge_vec, pass_u = edges$pass, pass_d = edges$pass)
}

total_length <- sum(V(g)$length)

# Split into weakly-connected structural components and run riverconn per
# component, then recombine (see header identity).
comp <- components(g, mode = "weak")
weighted_num <- 0.0
per_component <- data.frame()
for (k in seq_len(comp$no)) {
  member_v <- which(comp$membership == k)
  sub <- induced_subgraph(g, member_v)
  L_k <- sum(V(sub)$length)
  if (vcount(sub) == 1L) {
    # A lone reach is trivially fully connected to itself: index = 1.
    idx_k <- 1.0
  } else {
    res <- index_calculation(
      sub,
      weight = "length",
      nodes_id = "name",
      index_type = "full",
      c_ij_flag = TRUE,
      B_ij_flag = FALSE,
      dir_fragmentation_type = "symmetric",
      param = 0.9
    )
    idx_k <- res$index
  }
  weighted_num <- weighted_num + idx_k * (L_k^2)
  per_component <- rbind(per_component, data.frame(
    component = k, n_reaches = vcount(sub), length_m = L_k, index = idx_k
  ))
}

dci_full <- weighted_num / (total_length^2)

result <- data.frame(
  riverconn_version = as.character(packageVersion("riverconn")),
  n_reaches = vcount(g),
  n_structural_edges = ecount(g),
  n_components = comp$no,
  total_length_m = total_length,
  riverconn_dci_0_1 = dci_full,
  riverconn_dci_pct = 100.0 * dci_full
)
write.csv(result, out_csv, row.names = FALSE)
cat("riverconn DCI (0-1):", dci_full, " => pct:", 100.0 * dci_full, "\n")
cat("components:", comp$no, " reaches:", vcount(g), " edges:", ecount(g), "\n")

plugin = "hybrid"
plugin_config = dict(
    tp_size=8,
    pp_size=1,
    sp_size=1,
    zero_stage=2,
    overlap_allgather=False,
)

plugin_ae = "hybrid"
plugin_config_ae = dict(
    tp_size=8,
    pp_size=1,
    sp_size=1,
    zero_stage=2,
    overlap_allgather=False,
)

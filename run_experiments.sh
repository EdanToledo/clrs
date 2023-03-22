# NO HINTS

python ./clrs/examples/run.py --processor_type deepsets --wandb_run_name "DEEP SETS NO HINTS GROUNDTRUTH" --checkpoint_model_name "deepsets_no_hints"

python ./clrs/examples/run.py --processor_type mpnn --wandb_run_name "MPNN NO HINTS GROUNDTRUTH" --checkpoint_model_name "mpnn_no_hints"

python ./clrs/examples/run.py --processor_type pgn --wandb_run_name "PGN NO HINTS GROUNDTRUTH" --checkpoint_model_name "pgn_no_hints"

python ./clrs/examples/run.py --processor_type gatv2 --wandb_run_name "GATv2 NO HINTS GROUNDTRUTH" --checkpoint_model_name "gatv2_no_hints"

# HINTS - DECODED ONLY

python ./clrs/examples/run.py --hint_mode decoded_only --processor_type deepsets --wandb_run_name "DEEP SETS GROUNDTRUTH DECODED ONLY" --checkpoint_model_name "deepsets_decoded_hints"

python ./clrs/examples/run.py --hint_mode decoded_only --processor_type mpnn --wandb_run_name "MPNN GROUNDTRUTH DECODED ONLY" --checkpoint_model_name "mpnn_decoded_hints"

python ./clrs/examples/run.py --hint_mode decoded_only --processor_type pgn --wandb_run_name "PGN GROUNDTRUTH DECODED ONLY" --checkpoint_model_name "pgn_decoded_hints"

python ./clrs/examples/run.py --hint_mode decoded_only --processor_type gatv2 --wandb_run_name "GATv2 GROUNDTRUTH DECODED ONLY" --checkpoint_model_name "gatv2_decoded_hints"

# HINTS - ENCODED_DECODED

python ./clrs/examples/run.py --hint_mode encoded_decoded --processor_type deepsets --wandb_run_name "DEEP SETS GROUNDTRUTH ENCODED-DECODED" --checkpoint_model_name "deepsets_encoded_hints"

python ./clrs/examples/run.py --hint_mode encoded_decoded --processor_type mpnn --wandb_run_name "MPNN GROUNDTRUTH ENCODED-DECODED" --checkpoint_model_name "mpnn_encoded_hints"

python ./clrs/examples/run.py --hint_mode encoded_decoded --processor_type pgn --wandb_run_name "PGN GROUNDTRUTH ENCODED-DECODED" --checkpoint_model_name "pgn_encoded_hints"

python ./clrs/examples/run.py --hint_mode encoded_decoded --processor_type gatv2 --wandb_run_name "GATv2 GROUNDTRUTH ENCODED-DECODED" --checkpoint_model_name "gatv2_encoded_hints"

source /home/edantoledo/tests/clrs/venv_clrs/bin/activate

# BASE
python /home/edantoledo/tests/clrs/clrs/examples/run.py --hint_teacher_forcing 0.0 --hint_mode none --processor_type mpnn --wandb_run_name mpnn_base

python /home/edantoledo/tests/clrs/clrs/examples/run.py --hint_teacher_forcing 0.0 --hint_mode none --processor_type gatv2 --wandb_run_name gatv2_base

python /home/edantoledo/tests/clrs/clrs/examples/run.py --hint_teacher_forcing 0.0 --hint_mode none --processor_type deepsets --wandb_run_name deepsets_base

python /home/edantoledo/tests/clrs/clrs/examples/run.py --hint_teacher_forcing 0.0 --hint_mode none --processor_type pgn --wandb_run_name pgn_base

# With Hints

python /home/edantoledo/tests/clrs/clrs/examples/run.py --hint_teacher_forcing 0.0 --hint_mode encoded_decoded --processor_type mpnn --wandb_run_name mpnn_base_w_hints

python /home/edantoledo/tests/clrs/clrs/examples/run.py --hint_teacher_forcing 0.0 --hint_mode encoded_decoded --processor_type gatv2 --wandb_run_name gatv2_base_w_hints

python /home/edantoledo/tests/clrs/clrs/examples/run.py --hint_teacher_forcing 0.0 --hint_mode encoded_decoded --processor_type deepsets --wandb_run_name deepsets_base_w_hints

python /home/edantoledo/tests/clrs/clrs/examples/run.py --hint_teacher_forcing 0.0 --hint_mode encoded_decoded --processor_type pgn --wandb_run_name pgn_base_w_hints

# With Teacher Forcing 0.3

python /home/edantoledo/tests/clrs/clrs/examples/run.py --hint_teacher_forcing 0.3 --hint_mode encoded_decoded --processor_type mpnn --wandb_run_name mpnn_w_tf_3

python /home/edantoledo/tests/clrs/clrs/examples/run.py --hint_teacher_forcing 0.3 --hint_mode encoded_decoded --processor_type gatv2 --wandb_run_name gatv2_w_tf_3

python /home/edantoledo/tests/clrs/clrs/examples/run.py --hint_teacher_forcing 0.3 --hint_mode encoded_decoded --processor_type deepsets --wandb_run_name deepsets_w_tf_3

python /home/edantoledo/tests/clrs/clrs/examples/run.py --hint_teacher_forcing 0.3 --hint_mode encoded_decoded --processor_type pgn --wandb_run_name pgn_w_tf_3

# With Teacher Forcing 0.7

python /home/edantoledo/tests/clrs/clrs/examples/run.py --hint_teacher_forcing 0.7 --hint_mode encoded_decoded --processor_type mpnn --wandb_run_name mpnn_w_tf_7

python /home/edantoledo/tests/clrs/clrs/examples/run.py --hint_teacher_forcing 0.7 --hint_mode encoded_decoded --processor_type gatv2 --wandb_run_name gatv2_w_tf_7

python /home/edantoledo/tests/clrs/clrs/examples/run.py --hint_teacher_forcing 0.7 --hint_mode encoded_decoded --processor_type deepsets --wandb_run_name deepsets_w_tf_7

python /home/edantoledo/tests/clrs/clrs/examples/run.py --hint_teacher_forcing 0.7 --hint_mode encoded_decoded --processor_type pgn --wandb_run_name pgn_w_tf_7


# With Teacher Forcing 1.0

python /home/edantoledo/tests/clrs/clrs/examples/run.py --hint_teacher_forcing 1.0 --hint_mode encoded_decoded --processor_type mpnn --wandb_run_name mpnn_w_tf_1

python /home/edantoledo/tests/clrs/clrs/examples/run.py --hint_teacher_forcing 1.0 --hint_mode encoded_decoded --processor_type gatv2 --wandb_run_name gatv2_w_tf_1

python /home/edantoledo/tests/clrs/clrs/examples/run.py --hint_teacher_forcing 1.0 --hint_mode encoded_decoded --processor_type deepsets --wandb_run_name deepsets_w_tf_1

python /home/edantoledo/tests/clrs/clrs/examples/run.py --hint_teacher_forcing 1.0 --hint_mode encoded_decoded --processor_type pgn --wandb_run_name pgn_w_tf_1



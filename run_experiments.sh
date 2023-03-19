# navigate to clrs folder first, 
# source /opt/anaconda3/etc/profile.d/conda.sh # for local
# source /opt/conda/etc/profile.d/conda.sh # for gpu 

# conda activate causal_gnn
conda activate gnn
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$PYTHONPATH:$(pwd)

# BASE
python clrs/examples/run.py --seed 1 --hint_teacher_forcing 0.0 --hint_mode none --processor_type mpnn --wandb_run_name mpnn_base_ic_star_step_1

python clrs/examples/run.py --seed 1 --hint_teacher_forcing 0.0 --hint_mode none --processor_type gatv2 --wandb_run_name gatv2_base_ic_star_step_1

python clrs/examples/run.py --seed 1 --hint_teacher_forcing 0.0 --hint_mode none --processor_type deepsets --wandb_run_name deepsets_base_ic_star_step_1

python clrs/examples/run.py --seed 1 --hint_teacher_forcing 0.0 --hint_mode none --processor_type pgn --wandb_run_name pgn_base_ic_star_step_1

# With Hints

python clrs/examples/run.py --seed 1 --hint_teacher_forcing 0.0 --hint_mode encoded_decoded --processor_type mpnn --wandb_run_name mpnn_base_w_hints_ic_star_step_1

python clrs/examples/run.py --seed 1 --hint_teacher_forcing 0.0 --hint_mode encoded_decoded --processor_type gatv2 --wandb_run_name gatv2_base_w_hints_ic_star_step_1

python clrs/examples/run.py --seed 1 --hint_teacher_forcing 0.0 --hint_mode encoded_decoded --processor_type deepsets --wandb_run_name deepsets_base_w_hints_ic_star_step_1

python clrs/examples/run.py --seed 1 --hint_teacher_forcing 0.0 --hint_mode encoded_decoded --processor_type pgn --wandb_run_name pgn_base_w_hints_ic_star_step_1

# With Teacher Forcing 0.3

python clrs/examples/run.py --seed 1 --hint_teacher_forcing 0.3 --hint_mode encoded_decoded --processor_type mpnn --wandb_run_name mpnn_w_tf_3_ic_star_step_1

python clrs/examples/run.py --seed 1 --hint_teacher_forcing 0.3 --hint_mode encoded_decoded --processor_type gatv2 --wandb_run_name gatv2_w_tf_3_ic_star_step_1

python clrs/examples/run.py --seed 1 --hint_teacher_forcing 0.3 --hint_mode encoded_decoded --processor_type deepsets --wandb_run_name deepsets_w_tf_3_ic_star_step_1

python clrs/examples/run.py --seed 1 --hint_teacher_forcing 0.3 --hint_mode encoded_decoded --processor_type pgn --wandb_run_name pgn_w_tf_3_ic_star_step_1

# With Teacher Forcing 0.7

python clrs/examples/run.py --seed 1 --hint_teacher_forcing 0.7 --hint_mode encoded_decoded --processor_type mpnn --wandb_run_name mpnn_w_tf_7_ic_star_step_1

python clrs/examples/run.py --seed 1 --hint_teacher_forcing 0.7 --hint_mode encoded_decoded --processor_type gatv2 --wandb_run_name gatv2_w_tf_7_ic_star_step_1

python clrs/examples/run.py --seed 1 --hint_teacher_forcing 0.7 --hint_mode encoded_decoded --processor_type deepsets --wandb_run_name deepsets_w_tf_7_ic_star_step_1

python clrs/examples/run.py --seed 1 --hint_teacher_forcing 0.7 --hint_mode encoded_decoded --processor_type pgn --wandb_run_name pgn_w_tf_7_ic_star_step_1


# With Teacher Forcing 1.0

python clrs/examples/run.py --seed 1 --hint_teacher_forcing 1.0 --hint_mode encoded_decoded --processor_type mpnn --wandb_run_name mpnn_w_tf_1_ic_star_step_1

python clrs/examples/run.py --seed 1 --hint_teacher_forcing 1.0 --hint_mode encoded_decoded --processor_type gatv2 --wandb_run_name gatv2_w_tf_1_ic_star_step_1

python clrs/examples/run.py --seed 1 --hint_teacher_forcing 1.0 --hint_mode encoded_decoded -processor_type deepsets --wandb_run_name deepsets_w_tf_1_ic_star_step_1

python clrs/examples/run.py --seed 1 --hint_teacher_forcing 1.0 --hint_mode encoded_decoded --processor_type pgn --wandb_run_name pgn_w_tf_1_ic_star_step_1



# With LSTM

python clrs/examples/run.py --seed 1 --hint_teacher_forcing 0.0 --hint_mode encoded_decoded --use_lstm --processor_type mpnn --wandb_run_name mpnn_w_lstm_ic_star_step_1

python clrs/examples/run.py --seed 1 --hint_teacher_forcing 0.0 --hint_mode encoded_decoded --use_lstm --processor_type gatv2 --wandb_run_name gatv2_w_lstm_ic_star_step_1

python clrs/examples/run.py --seed 1 --hint_teacher_forcing 0.0 --hint_mode encoded_decoded --use_lstm --processor_type deepsets --wandb_run_name deepsets_w_lstm_ic_star_step_1

python clrs/examples/run.py --seed 1 --hint_teacher_forcing 0.0 --hint_mode encoded_decoded --use_lstm --processor_type pgn --wandb_run_name pgn_w_lstm_ic_star_step_1





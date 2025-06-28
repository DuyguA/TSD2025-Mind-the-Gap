python3 custom_run.py \
	--model_name_or_path="../models/whisper_puncted_segmented" \
	--phase="puncted_segmented"\
	--output_dir="./whisper-puncted-segmented" \
	--per_device_train_batch_size="32" \
	--per_device_eval_batch_size="32" \
	--num_train_epochs="150" \
	--learning_rate="1e-4" \
	--warmup_steps="1250" \
	--logging_steps="25" \
	--dataloader_num_workers="8" \
	--overwrite_output_dir \


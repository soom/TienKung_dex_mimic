python scripts/pkl_to_npz.py \
    --robot c1 \
    --batch_dir dataset/pkl_c1 \
    --raw_output_dir dataset/npz_c1_raw/ \
    --output_dir dataset/npz_c1 \
    --add_transition \
    --recompute_velocities

# python scripts/pkl_to_npz.py \
#     --robot dex_evt \
#     --batch_dir dataset/pkl_dex \
#     --raw_output_dir dataset/npz_dex_raw/ \
#     --output_dir dataset/npz_dex \
#     --add_transition \
#     --recompute_velocities

# python scripts/pkl_to_npz.py \
#     --batch_dir dataset/pkl/ \
#     --raw_output_dir dataset/npz_pro_raw/ \
#     --output_dir dataset/npz_pro/ \
#     --add_transition \
#     --recompute_velocities \

python scripts/pkl_to_npz.py \
    --robot dex_evt \
    --batch_dir dataset/pkl_dex \
    --raw_output_dir dataset/npz_dex_raw/ \
    --output_dir dataset/npz_dex \
    --add_transition \
    --recompute_velocities
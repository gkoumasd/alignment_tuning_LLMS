
python3 write_sdf.py --input_file $1 --direction=caption
./m2v.sh
python3 text_text2mol_metric.py --input_file $1 --use_gt



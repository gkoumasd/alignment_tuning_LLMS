
python3 write_sdf.py --input_file $1
./m2v.sh
python3 mol_text2mol_metric.py --input_file $1



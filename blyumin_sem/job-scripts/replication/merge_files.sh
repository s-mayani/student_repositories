#!/bin/bash
#SBATCH --partition=hourly
#SBATCH --time=00:10:00
#SBATCH --job-name=merge-files
#SBATCH --output=merge-files-%j.out
#SBATCH --error=merge-files-%j.err
#SBATCH --ntasks=1
#SBATCH --mem=8G

#cat lsf_512.*.csv > Data.csv

INPUT_DIR="/data/user/bobrov_e/ICs"
OUTPUT_DIR="/data/user/bobrov_e/data/lsf_512"
OUTPUT_FILE="$OUTPUT_DIR/Data.csv"

mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_FILE"

# Uncomment if you are unsure that the files
# have been correctly generated

#echo "Checking each file before merging..."
#for f in "$INPUT_DIR"/lsf_512.*.csv; do
#  bad=$(awk -F',' 'NF != 7' "$f" | wc -l)
#  if [[ "$bad" -ne 0 ]]; then
#    echo "$f has $bad malformed lines"
#  else
#    echo "$f is well-formed"
#  fi
#done

echo "Merging files from $INPUT_DIR..."
for f in "$INPUT_DIR"/lsf_512.*.csv; do
  # Add newline if the file doesn't end with one
  last_char=$(tail -c1 "$f" | od -An -t uC)
  if [[ "$last_char" -ne 10 ]]; then
    cat "$f"
    echo    # add newline if needed
  else
    cat "$f"
  fi
done > "$OUTPUT_FILE"

echo "Checking merged $OUTPUT_FILE..."
bad_lines=$(awk -F',' 'NF != 7' "$OUTPUT_FILE" | wc -l)

if [[ "$bad_lines" -eq 0 ]]; then
  echo "Merge successful: all lines are well-formed."
else
  echo "Merge failed: $bad_lines malformed lines found."
  exit 1
fi


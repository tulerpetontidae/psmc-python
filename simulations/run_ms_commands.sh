#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: ./run_ms_commands.sh <commands_file> <ms_path> <ms2psmcfa_path>"
    echo "commands_file: path to the file containing 'ms' commands"
    echo "ms_path: path to the 'ms' executable"
    echo "ms2psmcfa_path: path to the 'ms2psmcfa.py' script"
    exit 1
fi

# Parse command line arguments
while getopts ":c:m:p:" opt; do
  case $opt in
    c) cmd_file="$OPTARG"
    ;;
    m) ms_path="$OPTARG"
    ;;
    p) ms2psmcfa_path="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# Check if required arguments are supplied
if [ -z "$cmd_file" ] || [ -z "$ms_path" ] || [ -z "$ms2psmcfa_path" ]
then
  echo "Usage: ./run_ms_commands.sh -c <command_file> -m <ms_path> -p <ms2psmcfa_path>"
  exit 1
fi

# Check if command file exists
if [ ! -f "$cmd_file" ]
then
  echo "Error: Command file '$cmd_file' does not exist"
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p ms_output

# Run ms commands and save output to files
i=1
while read cmd; do
  output_file="ms_output/sim-$i.ms"
  ${ms_path} ${cmd#ms } > ${output_file}
  i=$((i+1))
done < "$cmd_file"

# Convert ms files to psmcfa format
for file in ms_output/*.ms; do
  output_file="${file%.*}.psmcfa"
  $ms2psmcfa_path $file > $output_file
done

echo "Done"

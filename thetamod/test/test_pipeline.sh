#!/usr/bin/env bash

FR3_subjects=('R1163T' 'R1166D' 'R1195E' 'R1200T' 'R1201P' 'R1202M')
PS4_FR_subjects=(R1299T R1302M)
FR50_subjects=(R1330D R1345D R1384J)
FR51_subjects=(R1275D R1308T R1321M R1339D R1351M R1384J)
FR6_subjects=(R1397D R1409D)

test_subjects(){
    experiment=$1
    shift
    session=$1
    shift
    while (( $# ))
    do
    subject=$1
    echo "${subject}"
    echo "$subject  " >> test_results.txt
    python ~/thetamod/run_pipeline.py -c -x $experiment -e 0 -s $subject -r /Volumes/rhino_root >> test_results.txt
    shift
    done
}

echo '' > test_results.txt

test_subjects FR5 1 "${FR51_subjects[@]}"
test_subjects "FR3" 0"${FR3_subjects[@]}"
test_subjects PS4_FR 0"${PS4_FR_subjects[@]}"
test_subjects FR5 0 "${FR50_subjects[@]}"
#test_subjects FR6 "${FR6_subjects[@]}" # FR6 is no good

### This reads in a 2-column file formated as '<number of occurences> <name of micrograph>'
### It will then return *only* <number> occurences from a second file
### This allows for a crude, but effective symbreak matcher
### Recommended to shuf the input file to prevent potential preferred orientations
### due to artifacts from the input order

while IFS=" " read my_num my_name remainder
do
	grep -m $my_num $my_name symbreak_not_class01.body.shuf >> symbreak_matched.star
done < this_many_class01.txt.reformat


### See:
# stackoverflow.com/questions/5013151/
### and also
# unix.stackexchange.com/questions/260840/

### Note:
# for those items with occurances > 1/2 of all potential occurances, there will still be an imbalance
### See: 
#	cat this_many_class01_column1.txt | sort | uniq -c | sort -k2h | less

### To monitor progress, try:
#	watch -d -n1 'wc -l symbreak_class01.star symbreak_matched.star'

#!/bin/bash

result=""

for i in {1..4}
do
    result+="Value$i "  # Concatenate the current value of i to the result string
done

echo "Concatenated result: $result"
